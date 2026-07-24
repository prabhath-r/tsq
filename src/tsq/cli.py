# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib.resources import as_file, files
from math import isfinite
from pathlib import Path
from typing import Any, Iterator

from .capacity import (
    DEFAULT_STATE_LIMIT,
    CapacityAnalysisLimitError,
    analyze_sustained_capacity,
    concept_target,
    topic_target,
)
from .corpus import load_bundle, parse_bundle, read_and_parse, validate_bundle
from .authoring import (
    AuthoringJobs,
    CoveragePlanner,
    QuarantineReviewQueue,
    deterministic_test_pipeline,
)
from .engine import AdaptiveEngine
from .evidence import (
    ACTION_PAYLOAD_CONTRACTS,
    ActionKind,
    ActionPhase,
    EvaluationStatus,
    ScorerKind,
)
from .errors import (
    ConflictError,
    ExhaustedError,
    NotFoundError,
    TSQError,
    ValidationError,
)
from .graph import KnowledgeGraph
from .quality import audit_corpus
from .performance import (
    ImportedCriterionResult,
    ImportedEvaluation,
    ScoringProviderRegistry,
    SyntheticDeterministicProvider,
)
from .performance_ledger import PerformanceLedger
from .performance_selection import recommend_performance_tasks
from .replay import ProjectionReplay, replay_or_error
from .store import Database, new_id


BUNDLED_CORPUS_PACKAGE = "tsq.data"
BUNDLED_CORPUS_NAME = "ai_curriculum.json"
BUNDLED_RELEASE_MARKER = "bundled_corpus_release"
BUNDLED_RESOURCE_DIGEST_MARKER = "bundled_corpus_resource_sha256"
# Releases shipped before the marker existed.  This narrow, immutable lineage
# lets `start` upgrade TSQ's own seed without silently replacing a user's
# explicitly imported active corpus.
LEGACY_BUNDLED_RELEASE_HASHES = frozenset(
    {
        "74d0f933a60dbaf3f58be5cdb571f7ea9df78277100363c6dce66257094fd4b7",
        "c0b297a2d3892d2ec9e98da3e2245cde3b5279615f1732f02e2b83bd795dce68",
        "eaa9973afee33293ed0294616bc234e30d0b4f4c609ac8a5e5542c252e518666",
        "4dbe4dc9a34e993554cbee7db90c074d23c2fb993081b077ec716ef8f50ee045",
        "0b140a5b8907c5523b89947815f51765231f7f5568c46585d22e881ffc6d2b9b",
        "871e96c45280bfa2f5400496b1ce60277fe5b376b30c39a816a665857fabbb1f",
        "7cc5b3507906ae3cb2216f2185144be3fbcd0fc6ad3ca4e411e39ca91e726637",
    }
)
BUNDLED_CORPUS_LABEL = f"{BUNDLED_CORPUS_PACKAGE}:{BUNDLED_CORPUS_NAME}"


@dataclass(frozen=True, slots=True)
class StarterCorpusStatus:
    """Outcome of a conservative bundled-corpus installation attempt."""

    installed: bool
    retained_release_id: str | None = None
    conflict: str | None = None
    legacy_generated_revocations: int = 0

    def __bool__(self) -> bool:
        return self.installed


@contextmanager
def _corpus_path(path: Path | None) -> Iterator[tuple[Path, str]]:
    """Resolve an explicit corpus or materialize the installed seed resource.

    ``importlib.resources.as_file`` also works for zipped importers, so the CLI
    does not depend on the source checkout layout after installation.
    """
    if path is not None:
        resolved = Path(path)
        yield resolved, str(resolved)
        return
    resource = files(BUNDLED_CORPUS_PACKAGE).joinpath(BUNDLED_CORPUS_NAME)
    with as_file(resource) as resolved:
        yield resolved, BUNDLED_CORPUS_LABEL


def _default_database() -> Path:
    configured = os.environ.get("TSQ_DB")
    return Path(configured) if configured else Path.cwd() / "tsq.db"


def _database(args: argparse.Namespace, *, require_corpus: bool = True) -> Database:
    database = Database(args.db)
    database.initialize()
    # Initialization may create or explicitly migrate a database.  Every
    # writable command then crosses the same exact current-schema boundary as
    # an inspection command before it can mutate application state.
    Database(database.path, read_only=True).validate_current_schema()
    if require_corpus:
        with database.read() as connection:
            active = connection.execute(
                "SELECT value FROM meta WHERE key = 'active_corpus_release'"
            ).fetchone()
        if not active:
            raise TSQError("The database has no active corpus. Run `tsq init` first.")
    return database


def _inspection_database(
    args: argparse.Namespace, *, require_corpus: bool = True
) -> Database:
    """Open a current database without migrating or changing learner state."""

    database = Database(args.db, read_only=True)
    database.validate_current_schema()
    if require_corpus:
        with database.read() as connection:
            active = connection.execute(
                """SELECT release.id
                   FROM meta
                   JOIN corpus_releases release
                     ON release.id = meta.value
                   WHERE meta.key = 'active_corpus_release'"""
            ).fetchone()
        if not active:
            raise TSQError(
                "The database has no valid active corpus. Run `tsq init` first."
            )
    return database


def _json_default(value: Any):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"Cannot encode {type(value).__name__}")


def _emit(value: Any, *, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(value, indent=2, sort_keys=True, default=_json_default))
    else:
        print(value)


def command_init(args: argparse.Namespace) -> None:
    database = _database(args, require_corpus=False)
    with _corpus_path(args.corpus) as (corpus_path, corpus_label):
        counts = database.import_corpus(
            *read_and_parse(corpus_path, include_catalog=True)
        )
    if args.json:
        _emit(
            {"database": str(database.path), "corpus": corpus_label, **counts},
            as_json=True,
        )
    else:
        print(f"Initialized {database.path}")
        print(
            f"Imported {counts['questions']} questions, {counts['concepts']} concepts, "
            f"and {counts['misconceptions']} misconception hypotheses."
        )
        if counts["legacy_generated_revocations"]:
            print(
                "Emergency-quarantined "
                f"{counts['legacy_generated_revocations']} unreviewed generated "
                "question(s) that were active in older releases."
            )


def _ensure_starter_corpus(database: Database) -> StarterCorpusStatus:
    """Install or safely advance TSQ's own bundled corpus lineage."""
    with database.read() as connection:
        active = connection.execute(
            "SELECT value FROM meta WHERE key = 'active_corpus_release'"
        ).fetchone()
        active_release_id = active["value"] if active else None
        release = (
            connection.execute(
                "SELECT bundle_hash FROM corpus_releases WHERE id = ?",
                (active_release_id,),
            ).fetchone()
            if active_release_id
            else None
        )
        marker = connection.execute(
            "SELECT value FROM meta WHERE key = ?",
            (BUNDLED_RELEASE_MARKER,),
        ).fetchone()
        resource_digest_marker = connection.execute(
            "SELECT value FROM meta WHERE key = ?",
            (BUNDLED_RESOURCE_DIGEST_MARKER,),
        ).fetchone()
    trusted_lineage = bool(
        active_release_id
        and (
            (marker and marker["value"] == active_release_id)
            or (
                release
                and release["bundle_hash"] in LEGACY_BUNDLED_RELEASE_HASHES
            )
        )
    )
    if active and not trusted_lineage:
        return StarterCorpusStatus(False)
    try:
        with _corpus_path(None) as (corpus_path, _):
            bundled_resource_digest = hashlib.sha256(
                corpus_path.read_bytes()
            ).hexdigest()
            if (
                active_release_id is not None
                and marker
                and marker["value"] == active_release_id
                and resource_digest_marker
                and resource_digest_marker["value"]
                == bundled_resource_digest
            ):
                return StarterCorpusStatus(False)
            imported = database.import_corpus(
                *read_and_parse(corpus_path, include_catalog=True)
            )
    except ConflictError as exc:
        # A short-lived historical bundled release accidentally changed an
        # existing source record.  Its users must still be able to study from
        # the valid release already sealed in their database.  Never rewrite
        # the registry or pretend that the new release was installed: retain
        # the immutable active snapshot and make the withheld upgrade visible.
        if not trusted_lineage or active_release_id is None:
            raise
        return StarterCorpusStatus(
            False,
            retained_release_id=active_release_id,
            conflict=str(exc),
        )
    installed_release_id = str(imported["release_id"])
    with database.transaction() as connection:
        connection.executemany(
            """INSERT INTO meta(key, value) VALUES (?, ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
            (
                (BUNDLED_RELEASE_MARKER, installed_release_id),
                (
                    BUNDLED_RESOURCE_DIGEST_MARKER,
                    bundled_resource_digest,
                ),
            ),
        )
    return StarterCorpusStatus(
        installed_release_id != active_release_id,
        legacy_generated_revocations=int(
            imported["legacy_generated_revocations"]
        ),
    )


def _starter_topic(database: Database, requested: str | None) -> str:
    if requested:
        return requested
    catalog = database.get_catalog()
    if not catalog["topics"]:
        concepts = database.get_graph().concepts
        return sorted(concepts)[0]
    default_id = (
        "t_large_language_models"
        if any(
            topic["id"] == "t_large_language_models"
            for topic in catalog["topics"]
        )
        else catalog["topics"][0]["id"]
    )
    if not sys.stdin.isatty():
        return default_id
    choices = sorted(
        catalog["topics"],
        key=lambda topic: (
            topic["id"] != default_id,
            topic["parent_id"] is not None,
            topic["sort_order"],
            topic["name"],
        ),
    )
    print("Choose a curriculum topic:")
    for index, topic in enumerate(choices, start=1):
        label = " (recommended)" if topic["id"] == default_id else ""
        print(
            f"  {index}. {topic['name']} — "
            f"{topic['direct_primary_questions']} direct questions{label}"
        )
    try:
        raw = input("topic [1]> ").strip()
    except EOFError:
        return default_id
    if not raw:
        return choices[0]["id"]
    if raw.isdigit() and 1 <= int(raw) <= len(choices):
        return choices[int(raw) - 1]["id"]
    return raw


def command_start(args: argparse.Namespace) -> None:
    database = _database(args, require_corpus=False)
    starter_status = _ensure_starter_corpus(database)
    topic = _starter_topic(database, args.topic)
    if starter_status:
        print("Installed the bundled reviewed curriculum catalog.")
    if starter_status.legacy_generated_revocations:
        print(
            "Emergency-quarantined "
            f"{starter_status.legacy_generated_revocations} unreviewed generated "
            "question(s) that were active in older releases."
        )
    if starter_status.conflict is not None:
        print(
            "Bundled curriculum update was withheld to preserve immutable "
            f"release {starter_status.retained_release_id}: "
            f"{starter_status.conflict} Continuing with that sealed release. "
            "To install the latest curriculum independently, run with a new "
            "database path such as `TSQ_DB=tsq-latest.db ./start`.",
            file=sys.stderr,
        )
    command_study(
        argparse.Namespace(
            db=args.db,
            learner=args.learner,
            name=args.name,
            topic=topic,
            mode=args.mode,
            limit=args.limit,
            seed=args.seed,
            ask_confidence=args.ask_confidence,
            explain_policy=args.explain_policy,
            details=args.details,
        )
    )


def command_import(args: argparse.Namespace) -> None:
    database = _database(args, require_corpus=False)
    counts = database.import_corpus(
        *read_and_parse(args.path, include_catalog=True)
    )
    _emit(counts, as_json=args.json)


def command_revoke(args: argparse.Namespace) -> None:
    database = _database(args)
    result = database.revoke_question(
        args.question,
        args.reason,
        idempotency_key=args.idempotency_key,
    )
    if args.json:
        _emit(result, as_json=True)
    else:
        state = "already revoked" if result["idempotent"] else "revoked"
        print(f"Question {result['question_id']} {state}: {result['reason']}")


def command_audit(args: argparse.Namespace) -> int:
    with _corpus_path(args.path) as (corpus_path, corpus_label):
        bundle = load_bundle(corpus_path, validate=False)
    structural_issues = validate_bundle(bundle)
    questions = []
    issues = list(structural_issues)
    if not any(issue.severity == "error" for issue in structural_issues):
        try:
            concepts, edges, misconceptions, _, questions = parse_bundle(bundle)
        except ValidationError as exc:
            issues.extend(exc.issues)
        else:
            issues = audit_corpus(
                questions,
                knowledge_graph=KnowledgeGraph(concepts, edges),
                misconceptions=misconceptions,
            )
    result = {
        "path": corpus_label,
        "questions": (
            len(bundle.get("questions", []))
            if isinstance(bundle, dict) and isinstance(bundle.get("questions"), list)
            else len(questions)
        ),
        "errors": [asdict(issue) for issue in issues if issue.severity == "error"],
        "warnings": [asdict(issue) for issue in issues if issue.severity == "warning"],
    }
    if args.json:
        _emit(result, as_json=True)
    else:
        print(f"{result['questions']} questions: {len(result['errors'])} errors, {len(result['warnings'])} warnings")
        for issue in issues:
            print(f"[{issue.severity.upper()}] {issue.question_id or 'corpus'} {issue.code}: {issue.message}")
    return 2 if result["errors"] or (args.strict and result["warnings"]) else 0


def command_capacity(args: argparse.Namespace) -> int:
    with _corpus_path(args.path) as (corpus_path, corpus_label):
        (
            concepts,
            edges,
            misconceptions,
            _,
            questions,
            _,
            topics,
        ) = read_and_parse(corpus_path, include_catalog=True)
    try:
        if args.concept:
            targets = (
                concept_target(
                    args.concept,
                    target_main_count=args.target_main_count,
                ),
            )
        elif args.topic:
            targets = (
                topic_target(
                    args.topic,
                    topics,
                    target_main_count=args.target_main_count,
                ),
            )
        else:
            if not topics:
                raise ValueError(
                    "The corpus has no curriculum topics; select --concept explicitly."
                )
            targets = tuple(
                topic_target(
                    topic.id,
                    topics,
                    target_main_count=args.target_main_count,
                )
                for topic in sorted(topics, key=lambda value: value.id)
            )
        report = analyze_sustained_capacity(
            questions,
            KnowledgeGraph(concepts, edges),
            misconceptions,
            targets,
            state_limit=args.state_limit,
        )
    except CapacityAnalysisLimitError as exc:
        raise ValidationError(
            "Capacity analysis is incomplete and no heuristic result was used: "
            + str(exc)
        ) from exc
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc

    strict_failures = [
        result.target.target_id
        for result in report.targets
        if result.status in {"blocked", "thin", "order_sensitive"}
    ]
    payload = {
        "path": corpus_label,
        "exact": True,
        "strict": bool(args.strict),
        "summary": {
            "target_count": len(report.targets),
            "requested_main_capacity": args.target_main_count,
            "strict_failure_count": len(strict_failures),
            "strict_failure_targets": strict_failures,
        },
        **report.to_dict(),
    }
    if args.json:
        _emit(payload, as_json=True)
    else:
        print(f"Exact sustained-capacity analysis: {corpus_label}")
        print(
            "  robust = safe under every eligible family order; "
            "achievable = best capacity-preserving order"
        )
        for result in report.targets:
            print(
                f"  {result.target.target_id:<34} "
                f"families={result.eligible_family_count:>3}  "
                f"robust={result.order_robust_main_capacity:>3}  "
                f"achievable={result.achievable_main_capacity:>3}  "
                f"concept-floor={result.owned_concept_order_robust_floor:>2}  "
                f"target={result.target_main_count:>2}  "
                f"order-loss={result.order_loss:>2}  {result.status}"
            )
        if len(report.targets) == 1:
            result = report.targets[0]
            reserve = result.maximum_capacity.terminal_main_family_ids
            print(
                f"  initial-safe={len(result.initial_safe_family_ids)}  "
                f"scope-concepts={len(result.scope_concept_ids)}  "
                f"terminal-reserve={len(reserve)}"
            )
            if result.aggregate_status != result.status:
                print(
                    f"  aggregate-status={result.aggregate_status}; "
                    f"owned-concept floor lowers final status to {result.status}"
                )
            concept_deficits = (
                ("missing", result.missing_owned_concept_ids),
                ("thin", result.thin_owned_concept_ids),
                ("order-sensitive", result.order_sensitive_owned_concept_ids),
            )
            for label, concept_ids in concept_deficits:
                if concept_ids:
                    print(f"  {label} concepts: " + ", ".join(concept_ids))
            if reserve:
                print("  reserve: " + ", ".join(reserve))
            for blocker in result.maximum_capacity.blockers[:5]:
                path = blocker.misconception_id or blocker.path_kind
                print(
                    f"  blocked: {blocker.family_id} / {path} "
                    f"({blocker.reason})"
                )
        if args.strict and strict_failures:
            print(
                "Strict capacity gate failed: " + ", ".join(strict_failures),
                file=sys.stderr,
            )
    return 2 if args.strict and strict_failures else 0


def command_topics(args: argparse.Namespace) -> None:
    database = _inspection_database(args)
    catalog = database.get_catalog()
    if catalog["topics"] and not getattr(args, "concepts", False):
        objective_count_by_concept: dict[str, int] = {}
        for objective in database.get_learning_objectives(catalog["release_id"]):
            objective_count_by_concept[objective.primary_concept_id] = (
                objective_count_by_concept.get(objective.primary_concept_id, 0) + 1
            )
        children: dict[str | None, list[dict[str, Any]]] = {}
        for topic in catalog["topics"]:
            children.setdefault(topic["parent_id"], []).append(topic)
        for siblings in children.values():
            siblings.sort(key=lambda item: (item["sort_order"], item["name"], item["id"]))

        rows: list[dict[str, Any]] = []

        def append_topic(
            topic: dict[str, Any], depth: int, path: list[str]
        ) -> tuple[int, int, int]:
            child_rows = children.get(topic["id"], [])
            descendant_questions = topic["direct_primary_questions"]
            descendant_concepts = len(topic["concepts"])
            descendant_learning_objectives = sum(
                objective_count_by_concept.get(concept["id"], 0)
                for concept in topic["concepts"]
            )
            row = {
                **topic,
                "depth": depth,
                "path": [*path, topic["name"]],
                "direct_concepts": len(topic["concepts"]),
                "direct_learning_objectives": descendant_learning_objectives,
            }
            rows.append(row)
            for child in child_rows:
                (
                    child_questions,
                    child_concepts,
                    child_learning_objectives,
                ) = append_topic(
                    child, depth + 1, row["path"]
                )
                descendant_questions += child_questions
                descendant_concepts += child_concepts
                descendant_learning_objectives += child_learning_objectives
            row["scope_primary_questions"] = descendant_questions
            row["scope_concepts"] = descendant_concepts
            row["scope_learning_objectives"] = descendant_learning_objectives
            return (
                descendant_questions,
                descendant_concepts,
                descendant_learning_objectives,
            )

        for domain in catalog["domains"]:
            for topic in children.get(None, []):
                if topic["domain_id"] == domain["id"]:
                    append_topic(topic, 0, [domain["name"]])
        payload = {
            "release_id": catalog["release_id"],
            "domains": catalog["domains"],
            "topics": rows,
        }
        if args.json:
            _emit(payload, as_json=True)
            return
        for domain in catalog["domains"]:
            print(domain["name"])
            for row in rows:
                if row["domain_id"] != domain["id"]:
                    continue
                indent = "  " * (row["depth"] + 1)
                concept_label = (
                    "concept" if row["scope_concepts"] == 1 else "concepts"
                )
                objective_label = (
                    "learning objective"
                    if row["scope_learning_objectives"] == 1
                    else "learning objectives"
                )
                print(
                    f"{indent}{row['name']} ({row['id']}) — "
                    f"{row['scope_primary_questions']} questions, "
                    f"{row['scope_concepts']} {concept_label}, "
                    f"{row['scope_learning_objectives']} {objective_label}"
                )
        return

    concepts = sorted(
        database.get_graph().concepts.values(),
        key=lambda concept: (concept.domain, concept.name),
    )
    with database.read() as connection:
        release_id = database.get_active_release_id(connection)
        counts = {
            row["concept_id"]: row["n"]
            for row in connection.execute(
                """SELECT qc.concept_id, COUNT(DISTINCT qc.question_id) AS n
                   FROM question_concepts qc
                   JOIN release_questions rq ON rq.question_id = qc.question_id
                   WHERE rq.release_id = ?
                     AND rq.status IN ('approved', 'calibrated')
                     AND qc.role = 'primary'
                     AND NOT EXISTS (
                         SELECT 1 FROM question_revocations revoked
                         WHERE revoked.question_id = qc.question_id
                     )
                   GROUP BY qc.concept_id""",
                (release_id,),
            ).fetchall()
        }
    rows = [
        {
            "id": concept.id,
            "name": concept.name,
            "domain": concept.domain,
            "direct_questions": counts.get(concept.id, 0),
        }
        for concept in concepts
    ]
    if args.json:
        _emit(rows, as_json=True)
    else:
        width = max(len(row["id"]) for row in rows)
        for row in rows:
            print(f"{row['id']:<{width}}  {row['direct_questions']:>2} items  {row['name']}")


def command_graph(args: argparse.Namespace) -> None:
    database = _inspection_database(args)
    graph = database.get_graph()
    try:
        topic = database.resolve_topic(args.topic)
    except NotFoundError:
        topic = None
    if topic is not None:
        root = topic["id"]
        root_name = topic["name"]
        target_type = "topic"
        release_id = topic["release_id"]
        scope = database.topic_scope(root, release_id)
    else:
        root = args.topic
        root_name = graph.concepts[root].name if root in graph.concepts else root
        target_type = "concept"
        release_id = None
        scope = graph.learning_scope(root)
    learning_objectives = database.get_learning_objectives(
        release_id, primary_concept_ids=scope
    )
    edges = [
        {
            "source": edge.source_id,
            "relation": edge.relation.value,
            "target": edge.target_id,
            "weight": edge.weight,
        }
        for edge in graph.edges
        if edge.source_id in scope and edge.target_id in scope
    ]
    result = {
        "root": root,
        "root_name": root_name,
        "target_type": target_type,
        "concepts": [
            {"id": concept_id, "name": graph.concepts[concept_id].name}
            for concept_id in sorted(scope)
        ],
        "learning_objectives": [
            {
                "id": objective.id,
                "name": objective.name,
                "primary_concept_id": objective.primary_concept_id,
                "operation": objective.operation.value,
                "evidence_type": objective.evidence_type,
            }
            for objective in learning_objectives
        ],
        "edges": edges,
    }
    if args.json:
        _emit(result, as_json=True)
    else:
        print(
            f"Learning scope for {root_name} ({root}): {len(scope)} graph concepts, "
            f"{len(learning_objectives)} learning objectives"
        )
        for objective in learning_objectives:
            print(
                f"  objective {objective.id}: {objective.name} "
                f"({objective.operation.value}; {objective.primary_concept_id})"
            )
        for edge in edges:
            print(f"  {edge['source']} --{edge['relation']}--> {edge['target']}")


def command_learner_add(args: argparse.Namespace) -> None:
    database = _database(args)
    learner = AdaptiveEngine(database).create_learner(args.learner_id, args.name)
    _emit(learner, as_json=args.json)


def command_session_start(args: argparse.Namespace) -> None:
    database = _database(args)
    engine = AdaptiveEngine(database)
    engine.create_learner(args.learner, args.name)
    session = engine.start_session(
        args.learner,
        args.topic,
        mode=args.mode,
        seed=args.seed,
        idempotency_key=args.idempotency_key,
    )
    _emit(session, as_json=args.json)


def command_session_end(args: argparse.Namespace) -> None:
    database = _database(args)
    session = AdaptiveEngine(database).end_session(
        args.session,
        status=args.status,
        reason=args.reason,
        idempotency_key=args.idempotency_key,
    )
    _emit(session, as_json=args.json)


def command_session_list(args: argparse.Namespace) -> None:
    if type(args.limit) is not int or not 1 <= args.limit <= 200:
        raise ValidationError("Session history limit must be from 1 to 200.")
    if args.learner is not None and not args.learner.strip():
        raise ValidationError("Session history learner filter must not be blank.")

    database = _inspection_database(args)
    filters: list[str] = []
    parameters: list[Any] = []
    if args.learner is not None:
        filters.append("session_row.learner_id = ?")
        parameters.append(args.learner)
    if args.status is not None:
        filters.append("session_row.status = ?")
        parameters.append(args.status)
    where = "WHERE " + " AND ".join(filters) if filters else ""
    parameters.append(args.limit)
    with database.read() as connection:
        rows = connection.execute(
            f"""WITH selected_sessions AS MATERIALIZED (
                    SELECT session_row.*
                    FROM sessions session_row
                    {where}
                    ORDER BY session_row.updated_at DESC, session_row.id DESC
                    LIMIT ?
                ), attempt_stats AS (
                    SELECT attempt.session_id,
                           COUNT(*) AS questions_answered,
                           SUM(attempt.is_correct) AS correct,
                           SUM(CASE WHEN attempt.selected_option_id IS NULL
                                    THEN 1 ELSE 0 END) AS abstained
                    FROM attempts attempt
                    JOIN selected_sessions selected
                      ON selected.id = attempt.session_id
                    GROUP BY attempt.session_id
                )
                SELECT session_row.id, session_row.learner_id,
                       learner.display_name,
                       session_row.corpus_release_id, session_row.topic_id,
                       session_row.root_concept_id,
                       COALESCE(topic.name, concept.name) AS target_name,
                       session_row.mode, session_row.phase, session_row.status,
                       session_row.step, session_row.created_at,
                       session_row.updated_at,
                       COALESCE(stats.questions_answered, 0) AS questions_answered,
                       COALESCE(stats.correct, 0) AS correct,
                       COALESCE(stats.abstained, 0) AS abstained
                FROM selected_sessions session_row
                JOIN learners learner ON learner.id = session_row.learner_id
                JOIN concepts concept ON concept.id = session_row.root_concept_id
                LEFT JOIN release_topics topic
                  ON topic.release_id = session_row.corpus_release_id
                 AND topic.topic_id = session_row.topic_id
                LEFT JOIN attempt_stats stats ON stats.session_id = session_row.id
                ORDER BY session_row.updated_at DESC, session_row.id DESC""",
            tuple(parameters),
        ).fetchall()
    sessions = []
    for row in rows:
        answered = int(row["questions_answered"])
        correct = int(row["correct"])
        abstained = int(row["abstained"])
        selected_answers = answered - abstained
        sessions.append(
            {
                "id": row["id"],
                "learner_id": row["learner_id"],
                "learner_name": row["display_name"],
                "corpus_release_id": row["corpus_release_id"],
                "topic_id": row["topic_id"],
                "root_concept_id": row["root_concept_id"],
                "target_name": row["target_name"],
                "mode": row["mode"],
                "phase": row["phase"],
                "status": row["status"],
                "step": int(row["step"]),
                "questions_answered": answered,
                "correct": correct,
                "abstained": abstained,
                "accuracy": correct / answered if answered else None,
                "selected_answers": selected_answers,
                "selected_incorrect": selected_answers - correct,
                "selected_accuracy": (
                    correct / selected_answers
                    if selected_answers
                    else None
                ),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )
    if args.json:
        _emit(sessions, as_json=True)
        return
    if not sessions:
        print("No sessions matched.")
        return
    for session in sessions:
        selected_summary = (
            f"{session['correct']}/{session['selected_answers']} selected "
            "correct"
            if session["selected_answers"]
            else "0 selected answers"
        )
        print(
            f"{session['id']} [{session['status']}] {session['learner_id']} · "
            f"{session['target_name']} · {session['mode']} · "
            f"{session['questions_answered']} completed · "
            f"{selected_summary} · {session['abstained']} skipped · "
            f"updated {session['updated_at']}"
        )


def _print_productive_shadow(summary: dict[str, Any]) -> None:
    if not summary["attempt_count"]:
        return
    behavior = summary["behavior"]
    observations = summary["rubric_observations"]
    statuses = ", ".join(
        f"{status} {count}"
        for status, count in summary["attempt_statuses"].items()
    )
    print("  productive-task shadow observations (diagnostic only):")
    print(
        f"    {summary['attempt_count']} attempt(s) across "
        f"{summary['distinct_task_count']} task(s) and "
        f"{summary['observed_task_families']} task family/families · "
        f"{summary['observed_elapsed_seconds']:.1f}s observed elapsed"
    )
    if statuses:
        print(f"    attempt status: {statuses}")
    print(
        f"    {behavior['actions']} semantic action(s) · "
        f"{behavior['hint_requests']} hint request(s) · "
        f"{behavior['check_runs']} check run(s) · "
        f"{behavior['answer_revisions']} answer revision(s)"
    )
    rubric_line = (
        f"    {observations['evaluations']} evaluation(s) · "
        f"{observations['criteria_observed']} rubric observation(s)"
    )
    if observations["valid_score_average"] is not None:
        rubric_line += (
            " · raw valid-score mean "
            f"{observations['valid_score_average'] * 100:.1f}%"
        )
    print(rubric_line)
    objective_ids = summary["scope_binding"]["objective_ids"]
    if objective_ids:
        rendered_objectives = ", ".join(objective_ids[:5])
        if len(objective_ids) > 5:
            rendered_objectives += f" (+{len(objective_ids) - 5} more)"
        print(
            "    explicit rubric objective binding(s): "
            + rendered_objectives
        )
    if observations["misconception_signals"]:
        signals = list(observations["misconception_signals"].items())
        rendered_signals = ", ".join(
            f"{misconception_id} {count}"
            for misconception_id, count in signals[:5]
        )
        if len(signals) > 5:
            rendered_signals += f" (+{len(signals) - 5} more)"
        print(
            "    rubric-reported named misconception signal(s): "
            + rendered_signals
        )
    print(
        "    boundary: no mastery update, certification claim, or adaptive "
        "routing effect"
    )


def command_session_report(args: argparse.Namespace) -> None:
    report = AdaptiveEngine(_inspection_database(args)).session_report(args.session)
    if args.json:
        _emit(report, as_json=True)
        return
    topic = report["topic"]
    target = topic["name"] if topic else report["root_concept_id"]
    response = report["response_time"]
    difficulty = report["difficulty"]
    print(f"Session {report['session_id']} · {target} · {report['status']}")
    inference = report["selected_response_inference"]
    print(
        "  inference boundary: provisional selected-response model; "
        "not empirically validated"
    )
    print(
        "    release-wide calibrated eligible items "
        f"{inference['calibrated_question_count']}/"
        f"{inference['eligible_question_count']} · numerical guard covers "
        "approximation only"
    )
    print(
        f"  {report['questions_answered']} completed · "
        f"{report['correct']} correct · "
        f"{report['selected_incorrect']} incorrect · "
        f"{report['abstained']} skipped"
    )
    if report["selected_answers"]:
        print(
            "    among selected answers: "
            f"{report['correct']}/{report['selected_answers']} correct "
            f"({report['selected_accuracy'] * 100:.1f}%)"
        )
    print(
        f"  active response time {response['active_seconds']:.1f}s · "
        f"wall time {response['wall_seconds']:.1f}s"
    )
    if response["selection_window_inconsistencies"]:
        print(
            "  timing warning: "
            f"{response['selection_window_inconsistencies']} submitted value(s) "
            "exceeded their selection-to-answer window"
        )
    if response["missing_values"]:
        print(
            f"  timing incomplete: {response['missing_values']} answer(s) had no "
            "submitted response duration"
        )
    if difficulty["average"] is not None:
        print(
            f"  authored difficulty {difficulty['average']:+.2f} average "
            f"({difficulty['minimum']:+.2f} to {difficulty['maximum']:+.2f})"
        )
    continuity = report["continuity"]
    if continuity["average_score"] is not None:
        print(
            f"  continuity {continuity['average_score']:.2f} average · "
            f"{report['exploration']['questions']} deliberate exploration probe(s) · "
            f"{report['remediation_questions']} targeted-practice/"
            "transfer-check question(s)"
        )
    if report["topic_distribution"]:
        rendered = ", ".join(
            f"{row['name']} {row['n']}" for row in report["topic_distribution"]
        )
        print(f"  topic mix: {rendered}")
    position_shadow = report["response_position_shadow"]
    position_evidence = position_shadow["evidence"]
    position_inference = position_shadow["inference"]
    position_status = position_inference["status"]
    analyzed_position_answers = position_evidence[
        "analyzed_non_abstained_observations"
    ]
    total_position_answers = position_evidence[
        "total_non_abstained_observations"
    ]
    position_window_suffix = (
        f", most recent of {total_position_answers}"
        if analyzed_position_answers < total_position_answers
        else ""
    )
    if position_status == "position_concentration_signal":
        dominant = position_inference["dominant_position"]
        adjusted = dominant["bonferroni_adjusted_probability"]["value"]
        print(
            "  response-position shadow: concentration signal at displayed "
            f"position {dominant['display_position']} "
            f"({dominant['selected_count']}/"
            f"{analyzed_position_answers}{position_window_suffix}; "
            f"family-wise p={adjusted:.3g})"
        )
    elif position_status == "no_signal":
        print(
            "  response-position shadow: no family-wise concentration signal "
            f"across {analyzed_position_answers} non-abstained answers"
            f"{position_window_suffix}"
        )
    elif position_status == "inconclusive":
        minimum = position_shadow["test_contract"][
            "minimum_non_abstained_observations"
        ]
        print(
            "  response-position shadow: inconclusive "
            f"({analyzed_position_answers}/{minimum} "
            "required non-abstained answers)"
        )
    else:
        print(
            "  response-position shadow: unavailable because its immutable "
            "evidence boundary is invalid"
        )
    print(
        "    shadow-only behavioral hypothesis; does not certify skill or "
        "change mastery or selection"
    )
    if report["objective_performance"]:
        print("  fine-grained selected-response evidence:")
        for objective in report["objective_performance"]:
            observed = objective["session"]
            projection = objective["current_projection"]
            selection_summary = (
                f"{observed['correct']}/{observed['selected_answers']} "
                "selected correct"
                if observed["selected_answers"]
                else "0 selected answers"
            )
            skipped = (
                f" · {observed['abstained']} skipped"
                if observed["abstained"]
                else ""
            )
            flags = (
                ", ".join(
                    reason.replace("_", " ")
                    for reason in objective["attention_reasons"]
                )
                or "no current warning"
            )
            print(
                f"    {objective['name']} ({objective['operation']}): "
                f"{selection_summary}{skipped} · "
                f"mastery {projection['mastery_probability'] * 100:.1f}% · "
                f"uncertainty {projection['uncertainty']:.2f} · "
                f"{projection['independent_families']} independent family/families"
            )
            print(f"      evidence note: {flags}")
    routing = report["adaptive_routing"]
    if routing["prerequisite_descents"] or routing["bounded_exits"]:
        print(
            f"  adaptive routing: {routing['prerequisite_descents']} prerequisite "
            f"descent(s) · {routing['prerequisite_resumptions']} resumed parent(s) · "
            f"{routing['prevented_reopenings']} verified boundary reopening(s) prevented"
        )
    if report["diagnostic_findings"]:
        print("  evidence-backed learning boundaries:")
        for finding in report["diagnostic_findings"][:5]:
            projection = finding["current_projection"]
            reasons = ", ".join(
                reason.replace("_", " ")
                for reason in finding["attention_reasons"]
            )
            print(
                f"    {finding['name']}: graph readiness "
                f"{projection['effective_readiness'] * 100:.1f}% · {reasons}"
            )
    _print_productive_shadow(report["productive_skill_shadow"])
    print(
        "  difficulty values are authored priors; no empirical calibration has "
        "been validated."
    )


def _presentation_dict(presentation) -> dict[str, Any]:
    result = {
        "decision_id": presentation.decision_id,
        "session_id": presentation.session_id,
        "phase": presentation.phase.value,
        "question_id": presentation.question.id,
        "family_id": presentation.question.family_id,
        "kind": presentation.question.kind.value,
        "pedagogical_role": presentation.pedagogical_role,
        "stem": presentation.question.stem,
        "options": [
            {"id": option.id, "text": option.text} for option in presentation.ordered_options
        ],
        "selection": {
            "rationale": presentation.rationale,
            "propensity": presentation.propensity,
            "score": presentation.score.terms(),
        },
    }
    if presentation.question.objective is not None:
        objective = presentation.question.objective
        result["learning_objective"] = {
            "id": objective.id,
            "name": objective.name,
            "description": objective.description,
            "operation": objective.operation.value,
            "evidence_type": objective.evidence_type,
        }
    return result


def command_next(args: argparse.Namespace) -> None:
    database = _database(args)
    presentation = AdaptiveEngine(database).next_question(args.session)
    result = _presentation_dict(presentation)
    if args.json:
        _emit(result, as_json=True)
    else:
        print(f"[{result['phase']}] {result['stem']}")
        if "learning_objective" in result:
            objective = result["learning_objective"]
            print(
                f"Objective: {objective['name']} ({objective['id']}) · "
                f"{objective['operation']} / {objective['evidence_type']}"
            )
        for index, option in enumerate(result["options"], start=1):
            print(f"  {index}. {option['text']}  (id: {option['id']})")
        if args.explain:
            print(f"Why: {result['selection']['rationale']}")
        print(f"Decision: {result['decision_id']}")


def _submission_dict(
    result, objective_names: dict[str, str] | None = None
) -> dict[str, Any]:
    outcome = (
        "abstained"
        if result.selected_option is None
        else "correct" if result.correct else "incorrect"
    )
    payload = {
        "interaction_id": result.interaction_id,
        "correct": result.correct,
        "outcome": outcome,
        "selected_option_id": result.selected_option.id if result.selected_option else None,
        "correct_option_id": result.correct_option.id,
        "selected_rationale": result.selected_option.rationale if result.selected_option else None,
        "correct_rationale": result.correct_option.rationale,
        "next_phase": result.next_phase.value,
        "focus_concept_id": result.focus_concept_id,
        "focus_misconception_id": result.focus_misconception_id,
        "transition_reason": result.transition_reason,
        "boundary_decision": result.boundary_decision,
        "state_changes": list(result.state_changes),
        "idempotent_replay": result.idempotent_replay,
    }
    objective_names = objective_names or {}
    assessed_objective_ids = tuple(
        dict.fromkeys(
            change["objective_id"]
            for change in result.state_changes
            if "objective_id" in change
        )
    )
    if assessed_objective_ids:
        objective_id = assessed_objective_ids[0]
        state_change = next(
            change
            for change in result.state_changes
            if change.get("objective_id") == objective_id
        )
        payload["learning_objective"] = {
            "id": objective_id,
            "name": objective_names.get(objective_id, objective_id),
            "state_change": dict(state_change),
        }
    if result.focus_objective_id is not None:
        payload["focus_objective_id"] = result.focus_objective_id
        payload["focus_learning_objective"] = {
            "id": result.focus_objective_id,
            "name": objective_names.get(
                result.focus_objective_id, result.focus_objective_id
            ),
        }
    return payload


def _decision_objective_names(
    database: Database, decision_id: str
) -> dict[str, str]:
    with database.read() as connection:
        row = connection.execute(
            "SELECT corpus_release_id FROM decisions WHERE id = ?",
            (decision_id,),
        ).fetchone()
    if row is None:
        return {}
    return {
        objective.id: objective.name
        for objective in database.get_learning_objectives(
            row["corpus_release_id"]
        )
    }


def _record_cli_feedback(
    engine: AdaptiveEngine,
    *,
    decision_id: str,
    selected_option_id: str | None,
    correct_option_id: str,
    selected_rationale: str | None,
    correct_rationale: str,
) -> None:
    """Durably record feedback only after it reaches the CLI output boundary."""

    material = json.dumps(
        {
            "decision_id": decision_id,
            "selected_option_id": selected_option_id,
            "correct_option_id": correct_option_id,
            "selected_rationale": selected_rationale,
            "correct_rationale": correct_rationale,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    feedback_digest = hashlib.sha256(material).hexdigest()
    idempotency_digest = hashlib.sha256(
        f"{decision_id}:{feedback_digest}".encode("utf-8")
    ).hexdigest()
    engine.record_action(
        decision_id,
        "feedback_shown",
        {"feedback_digest": feedback_digest},
        stage="post_feedback",
        idempotency_key=f"cli-feedback:{idempotency_digest}",
    )


def command_answer(args: argparse.Namespace) -> None:
    database = _database(args)
    engine = AdaptiveEngine(database)
    option_id = None if args.option in {"?", "unknown", "none"} else args.option
    confidence = None if option_id is None else args.confidence
    try:
        result = engine.submit_answer(
            args.decision,
            option_id,
            confidence=confidence,
            response_ms=args.response_ms,
            hint_count=args.hints,
            feedback_shown=False,
            idempotency_key=args.idempotency_key,
        )
    except ConflictError as exc:
        legacy_abstention_retry = bool(
            option_id is None
            and args.confidence is not None
            and args.idempotency_key
            and str(exc)
            == "Idempotency key was reused with different answer inputs."
        )
        if not legacy_abstention_retry:
            raise
        # Older CLI versions persisted a supplied confidence with an
        # abstention. New submissions normalize it away, but an exact retry
        # must still reproduce the old immutable command payload.
        result = engine.submit_answer(
            args.decision,
            option_id,
            confidence=args.confidence,
            response_ms=args.response_ms,
            hint_count=args.hints,
            feedback_shown=False,
            idempotency_key=args.idempotency_key,
        )
    objective_names = _decision_objective_names(database, args.decision)
    payload = _submission_dict(result, objective_names)
    if args.json:
        _emit(payload, as_json=True)
    else:
        if result.selected_option is None:
            print("Skipped — you chose 'I do not know'.")
        else:
            print("Correct." if result.correct else "Not correct.")
        if result.selected_option:
            print(f"Your choice: {result.selected_option.rationale}")
        print(f"Answer: {result.correct_option.text}")
        print(result.correct_option.rationale)
        assessed = payload.get("learning_objective")
        if assessed:
            change = next(
                (
                    item
                    for item in result.state_changes
                    if item.get("objective_id") == assessed["id"]
                ),
                None,
            )
            movement = ""
            if change is not None:
                movement = (
                    f" · evidence projection "
                    f"{change['prior_mastery'] * 100:.1f}% → "
                    f"{change['posterior_mastery'] * 100:.1f}%"
                )
            print(
                f"Assessed objective: {assessed['name']} ({assessed['id']})"
                f"{movement}"
            )
        print(f"Next phase: {result.next_phase.value}")
        if result.focus_misconception_id:
            print(f"Current hypothesis: {result.focus_misconception_id}")
        focus_objective = payload.get("focus_learning_objective")
        if focus_objective:
            print(
                "Next probe objective: "
                f"{focus_objective['name']} ({focus_objective['id']})"
            )
        print(f"Adaptive path: {result.transition_reason}")
    _record_cli_feedback(
        engine,
        decision_id=args.decision,
        selected_option_id=(
            result.selected_option.id if result.selected_option else None
        ),
        correct_option_id=result.correct_option.id,
        selected_rationale=(
            result.selected_option.rationale if result.selected_option else None
        ),
        correct_rationale=result.correct_option.rationale,
    )


def _strict_action_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValidationError(
                f"Action payload contains duplicate field {key!r}."
            )
        payload[key] = value
    return payload


def _reject_action_constant(value: str) -> None:
    raise ValidationError(
        f"Action payload contains non-finite number {value!r}."
    )


def _action_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.payload_file is not None:
        try:
            if args.payload_file.stat().st_size > 16_384:
                raise ValidationError(
                    "Action payload files must not exceed 16384 bytes."
                )
            raw = args.payload_file.read_text(encoding="utf-8")
        except UnicodeError as exc:
            raise ValidationError(
                f"Action payload file {args.payload_file} is not valid UTF-8."
            ) from exc
        except OSError as exc:
            raise ValidationError(
                f"Could not read action payload file {args.payload_file}: {exc}"
            ) from exc
    else:
        raw = args.payload
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > 16_384:
        raise ValidationError("Action payload JSON must not exceed 16384 bytes.")
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_strict_action_object,
            parse_constant=_reject_action_constant,
        )
    except ValidationError:
        raise
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Action payload must be valid JSON: {exc}") from exc
    if type(payload) is not dict:
        raise ValidationError("Action payload JSON must be an object.")
    return payload


def command_action_record(args: argparse.Namespace) -> None:
    database = _database(args)
    artifact_values = (
        args.artifact_sha256,
        args.artifact_size_bytes,
        args.artifact_media_type,
    )
    if any(value is not None for value in artifact_values) and not all(
        value is not None for value in artifact_values
    ):
        raise ValidationError(
            "Artifact references require --artifact-sha256, "
            "--artifact-size-bytes, and --artifact-media-type together."
        )
    artifact = (
        {
            "sha256": args.artifact_sha256,
            "size_bytes": args.artifact_size_bytes,
            "media_type": args.artifact_media_type,
        }
        if all(value is not None for value in artifact_values)
        else None
    )
    action = AdaptiveEngine(database).record_action(
        args.decision,
        args.action_type,
        _action_payload(args),
        stage=args.stage,
        artifact=artifact,
        idempotency_key=args.idempotency_key,
    )
    if args.json:
        _emit(action, as_json=True)
        return
    replay = " (idempotent replay)" if action["idempotent_replay"] else ""
    print(
        f"Recorded {action['action_type']} action #{action['sequence']} "
        f"for {action['decision_id']}{replay}."
    )
    print(f"Action: {action['id']}")
    print(f"Event: {action['event_id']}")


def command_action_list(args: argparse.Namespace) -> None:
    database = _inspection_database(args)
    actions = AdaptiveEngine(database).list_actions(args.decision)
    if args.json:
        _emit(actions, as_json=True)
        return
    if not actions:
        print("No semantic actions recorded for this decision.")
        return
    for action in actions:
        print(
            f"{action['sequence']:>3}. {action['stage']:<13} "
            f"{action['action_type']:<24} {action['occurred_at']}"
        )
        print(f"     {json.dumps(action['payload'], sort_keys=True)}")


def command_action_kinds(args: argparse.Namespace) -> None:
    contracts = {
        kind.value: dict(ACTION_PAYLOAD_CONTRACTS[kind]) for kind in ActionKind
    }
    if args.json:
        _emit(contracts, as_json=True)
        return
    print("Exact semantic-action payload contracts:")
    for kind, fields in contracts.items():
        rendered = (
            ", ".join(f"{name}: {field_type}" for name, field_type in fields.items())
            if fields
            else "empty object"
        )
        print(f"  {kind:<24} {rendered}")
    print("Content is never accepted directly; content-bearing fields use SHA-256 digests.")


def command_profile(args: argparse.Namespace) -> None:
    database = _inspection_database(args)
    profile = AdaptiveEngine(database).profile(args.learner, root_concept_id=args.topic)
    if args.json:
        _emit(profile, as_json=True)
        return
    print(f"Learner: {profile['learner_id']}")
    inference = profile["selected_response_inference"]
    print(
        "Inference boundary: provisional selected-response model; "
        "not empirically validated"
    )
    print(
        "  release-wide calibrated eligible items "
        f"{inference['calibrated_question_count']}/"
        f"{inference['eligible_question_count']} · numerical guard covers "
        "approximation only"
    )
    objectives = profile.get("learning_objectives", [])
    assessed_objectives = [
        objective
        for objective in objectives
        if objective["evidence_mass"] > 0
    ]
    assessed = [skill for skill in profile["skills"] if skill["evidence_mass"] > 0]
    if not assessed and not assessed_objectives:
        print("No selected-response evidence yet.")
    if assessed_objectives:
        print("Assessed selected-response objectives:")
        for objective in sorted(
            assessed_objectives,
            key=lambda row: (row["mastery"], row["name"]),
        ):
            print(
                f"  P(competence≥65%)={objective['mastery'] * 100:5.1f}%  "
                f"expected={objective['expected_competence'] * 100:5.1f}%  "
                f"latent-σ={objective['uncertainty']:.2f}  "
                f"{objective['state']:<10}  {objective['name']} "
                f"({objective['objective_id']}; {objective['operation']}; "
                f"{objective['independent_families']} independent families, "
                f"{objective['delayed_retrievals']} delayed)"
            )
            if objective["mastery_probability_error_bound"] > 0.0:
                print(
                    "      exact-grid estimate "
                    f"{objective['estimated_mastery_probability'] * 100:.2f}% · "
                    "conservative numerical guard "
                    f"{objective['mastery_probability_error_bound'] * 100:.3f}% · "
                    f"model {objective['inference_model_version']}"
                )
        print(f"  Scope: {profile['objective_evidence_scope']}")
    if assessed and objectives:
        print(
            "Graph concept context (supporting projection, not productive-skill "
            "certification):"
        )
    for skill in sorted(assessed, key=lambda row: (row["mastery"], row["name"])):
        mastery_label = "P(derived floor)" if objectives else "P(mastered)"
        print(
            f"  {mastery_label}={skill['mastery'] * 100:5.1f}%  "
            f"expected={skill['expected_competence'] * 100:5.1f}%  "
            f"latent-σ={skill['uncertainty']:.2f}  "
            f"graph-ready={skill['effective_readiness'] * 100:5.1f}%  "
            f"{skill['state']:<10}  {skill['name']} "
            f"({skill['independent_families']} independent families, "
            f"{skill['operation_kinds']} operations, {skill['delayed_retrievals']} delayed)"
        )
        if skill["bottleneck_name"]:
            print(
                f"      prerequisite support {skill['prerequisite_support'] * 100:.1f}% · "
                f"current boundary {skill['bottleneck_name']}"
            )
    if profile["active_misconceptions"]:
        print("Active misconception hypotheses:")
        for item in profile["active_misconceptions"]:
            print(f"  {item['probability'] * 100:5.1f}%  {item['name']}")
    monitored = [
        item
        for item in profile.get("misconception_hypotheses", ())
        if item.get("status") == "monitoring"
    ]
    if monitored:
        print("Monitored misconception hypotheses (below routing threshold):")
        for item in monitored:
            print(f"  {item['probability'] * 100:5.1f}%  {item['name']}")
    _print_productive_shadow(profile["productive_skill_shadow"])


def command_trace(args: argparse.Namespace) -> None:
    database = _inspection_database(args)
    decisions = AdaptiveEngine(database).trace(args.session)
    if args.json:
        _emit(decisions, as_json=True)
        return
    for decision in reversed(decisions):
        score = decision["selected_score"]
        status = "answered" if decision["consumed_at"] else "pending"
        print(
            f"{decision['id']} [{decision['phase']}/{status}] {decision['question_id']} "
            f"score={score['total']:.3f} p(correct)={score['predicted_correct']:.2f}"
        )
        print(f"  {decision['rationale']}")


def command_coverage(args: argparse.Namespace) -> None:
    database = (
        _database(args) if args.enqueue else _inspection_database(args)
    )
    planner = CoveragePlanner(database)
    filters = {
        "topic": args.topic,
        "concept": args.concept,
        "objective": args.objective,
        "misconception": args.misconception,
        "kind": args.kind,
        "goal": args.goal,
        "maximum_difficulty": args.maximum_difficulty,
    }
    gaps = planner.gaps(
        limit=args.limit,
        topic_filter=args.topic,
        concept_filter=args.concept,
        objective_filter=args.objective,
        misconception_filter=args.misconception,
        kind_filter=args.kind,
        goal_filter=args.goal,
        maximum_difficulty=args.maximum_difficulty,
    )
    job_ids = planner.enqueue(gaps) if args.enqueue else []
    payload = {
        "gap_count": len(gaps),
        "filters": {
            key: value for key, value in filters.items() if value is not None
        },
        "enqueued_job_ids": job_ids,
        "gaps": [
            {
                "priority": gap.priority,
                "current_count": gap.current_count,
                "target_count": gap.target_count,
                "blueprint": asdict(gap.blueprint),
            }
            for gap in gaps
        ],
    }
    if args.json:
        _emit(payload, as_json=True)
        return
    qualifier = " matching the requested scope" if payload["filters"] else ""
    print(f"Top {len(gaps)} corpus coverage gaps{qualifier}")
    for gap in gaps:
        blueprint = gap.blueprint
        target = blueprint.learning_objective_id or blueprint.concept_id
        if blueprint.target_misconception_id:
            target = f"{target} / {blueprint.target_misconception_id}"
        print(
            f"  {target:<48} {blueprint.kind:<16} "
            f"{gap.current_count}/{gap.target_count}  difficulty={blueprint.target_difficulty:+.2f}"
        )
    if job_ids:
        print(f"Enqueued {len(job_ids)} quarantined authoring jobs.")


def command_jobs_list(args: argparse.Namespace) -> None:
    jobs = AuthoringJobs(_inspection_database(args)).list(
        status=args.status, limit=args.limit
    )
    if args.json:
        _emit(jobs, as_json=True)
        return
    if not jobs:
        print("No generation jobs matched.")
        return
    for job in jobs:
        blueprint = job["blueprint"]
        target = blueprint.get("learning_objective_id") or blueprint["concept_id"]
        if blueprint.get("target_misconception_id"):
            target = f"{target} / {blueprint['target_misconception_id']}"
        print(
            f"{job['id']} [{job['status']}] {target} / "
            f"{blueprint['kind']}  attempts={job['run_count']}"
        )


def command_jobs_show(args: argparse.Namespace) -> None:
    job = AuthoringJobs(_inspection_database(args)).show(args.job)
    if args.json:
        _emit(job, as_json=True)
        return
    blueprint = job["blueprint"]
    print(f"Job {job['id']} [{job['status']}]")
    print(
        f"  target: {blueprint['concept_id']} / {blueprint['kind']} "
        f"at difficulty {blueprint['target_difficulty']:+.2f}"
    )
    if blueprint.get("learning_objective_id"):
        print(
            "  objective: "
            f"{blueprint['learning_objective_name']} "
            f"({blueprint['learning_objective_id']}; "
            f"{blueprint['learning_objective_operation']})"
        )
    if blueprint.get("target_misconception_id"):
        print(
            "  exact misconception: "
            f"{blueprint['target_misconception_id']}"
        )
    if blueprint.get("corpus_release_id"):
        print(f"  pinned release: {blueprint['corpus_release_id']}")
    print(f"  coverage goal: {blueprint.get('coverage_goal', 'concept_kind')}")
    print(f"  sources: {', '.join(blueprint['source_ids'])}")
    print(f"  attempts: {job['run_count']}")
    for run in job["runs"]:
        finished = run["completed_at"] or "in progress"
        print(
            f"    {run['attempt']}: {run['status']} via "
            f"{run['provider']}/{run['model']} ({finished})"
        )
        if run.get("error"):
            error = run["error"]
            error_type = error.get("error_type", "provider_error")
            message = error.get("error", "No provider error message was recorded.")
            print(f"      error [{error_type}]: {message}")
        validation = run.get("validation")
        if validation is not None:
            issues = validation.get("deterministic_issues", ())
            error_count = sum(
                issue.get("severity") == "error" for issue in issues
            )
            warning_count = sum(
                issue.get("severity") == "warning" for issue in issues
            )
            review_count = len(validation.get("reviews", ()))
            print(
                "      validation: "
                f"{error_count} error(s), {warning_count} warning(s); "
                f"{review_count} independent review(s)"
            )
            for issue in issues:
                severity = issue.get("severity", "unknown")
                code = issue.get("code", "unspecified")
                message = issue.get(
                    "message",
                    "No validation detail was recorded.",
                )
                print(f"        [{severity}] {code}: {message}")
    if job["raw_output"] is not None:
        print(
            f"  artifact: {job['raw_output'].get('id', '<unknown>')} "
            f"[{job['raw_output'].get('status', '<missing>')}]"
        )
        print("  activation: none (reviewed artifacts remain quarantined)")
def _read_source_context(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValidationError(f"Could not read source context {path}: {exc}") from exc
    if len(raw) > 2 * 1024 * 1024:
        raise ValidationError("Source context exceeds the 2 MiB operational limit.")
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError(f"Source context {path} is not valid UTF-8.") from exc
    if not value.strip():
        raise ValidationError("Source context must contain approved source material.")
    return value


def command_jobs_run(args: argparse.Namespace) -> None:
    database = _database(args)
    if args.provider != "deterministic-test":
        raise ValidationError(f"Unsupported authoring provider: {args.provider!r}.")
    result = deterministic_test_pipeline(database).run_job(
        args.job, _read_source_context(args.source_context)
    )
    if args.json:
        _emit(result, as_json=True)
        return
    print(
        f"Job {result['job_id']} attempt {result['attempt']} finished "
        f"{result['status']}."
    )
    print(
        f"Artifact {result['item'].get('id', '<unknown>')} remains quarantined; "
        "no live question or corpus release was changed."
    )


def command_jobs_retry(args: argparse.Namespace) -> None:
    job = AuthoringJobs(_database(args)).retry(
        args.job, recover_running=args.recover_running
    )
    if args.json:
        _emit(job, as_json=True)
        return
    print(
        f"Job {job['id']} is planned for explicit retry; "
        f"{job['run_count']} immutable prior attempt(s) retained."
    )


def command_reviews_show(args: argparse.Namespace) -> None:
    result = AuthoringJobs(_inspection_database(args)).reviews(args.job)
    if args.json:
        _emit(result, as_json=True)
        return
    if not result["review_attempts"]:
        print(f"Generation job {args.job} has no completed independent reviews.")
        return
    print(f"Independent reviews for generation job {args.job}")
    for attempt in result["review_attempts"]:
        for review in attempt["reviews"]:
            reviewer = review.get("reviewer", {})
            output = review.get("output", {})
            print(
                f"  attempt {attempt['attempt']} "
                f"[run {attempt['status']}]: "
                f"{reviewer.get('reviewer_name', '<unknown>')} -> "
                f"{output.get('verdict', '<invalid>')}"
            )


def command_quarantine_list(args: argparse.Namespace) -> None:
    rows = QuarantineReviewQueue(_inspection_database(args)).list(
        topic=args.topic,
        concept_id=args.concept,
        learning_objective_id=args.objective,
        limit=args.limit,
    )
    if args.json:
        _emit(rows, as_json=True)
        return
    if not rows:
        print("No quarantined questions matched.")
        return
    for row in rows:
        target = (
            row["learning_objective_id"]
            or row["primary_concept_id"]
        )
        review_claim = row["provenance_claims"].get(
            "human_review_status"
        )
        print(
            f"{row['question_id']} "
            f"[{row['activation_ceiling']}/{row['kind']}] "
            f"difficulty={row['difficulty']:+.2f}  {target}"
        )
        print(
            f"  sources={row['source_count']} "
            f"recorded-reviews={row['recorded_item_review_count']} "
            f"human-review-claim={review_claim or 'unspecified'}"
        )
        if row["revoked"]:
            print(
                "  permanently revoked: "
                f"{row['revocation_reason']}"
            )
    print("All listed questions remain quarantined and runtime-ineligible.")


def command_quarantine_show(args: argparse.Namespace) -> None:
    detail = QuarantineReviewQueue(
        _inspection_database(args)
    ).show(args.question)
    if args.json:
        _emit(detail, as_json=True)
        return
    question = detail["question"]
    print(
        f"Question {question['id']} [{detail['activation_ceiling']}] "
        f"{question['kind']} difficulty={question['difficulty']:+.2f}"
    )
    if question["learning_objective_id"]:
        print(f"  objective: {question['learning_objective_id']}")
    print(f"  family: {question['family_id']}")
    print(f"  content: {detail['question_content_sha256']}")
    print(f"\n{question['stem']}")
    for option in question["options"]:
        marker = "best" if option["correct"] else "distractor"
        print(f"  {option['id']}. [{marker}] {option['text']}")
        if option["misconception_id"]:
            print(
                f"     misconception: {option['misconception_id']}"
            )
        print(f"     rationale: {option['rationale']}")
    print("Sources:")
    for source in detail["sources"]:
        locator = f" · {source['uri']}" if source["uri"] else ""
        print(f"  {source['id']}: {source['title']}{locator}")
    if detail["revocation"] is not None:
        print(
            "Revocation: this immutable question ID must not be promoted; "
            f"{detail['revocation']['reason']}"
        )
    print(
        "Activation: none; provenance review labels are inspection claims, "
        "not human authority."
    )


def command_quarantine_packet(args: argparse.Namespace) -> None:
    packet = QuarantineReviewQueue(
        _inspection_database(args)
    ).packet(args.question, stage=args.stage)
    if args.json:
        _emit(packet, as_json=True)
        return
    print(
        f"Review packet {packet['schema']} for "
        f"{packet['question_id']}"
    )
    print(f"  packet sha256: {packet['packet_sha256']}")
    print(f"  stage: {args.stage}")
    print(
        "  activation: none; use --json to export the content-bound "
        f"{args.stage} inspection packet"
    )


def command_task_import(args: argparse.Namespace) -> None:
    result = PerformanceLedger(_database(args)).import_release(args.path)
    if args.json:
        _emit(result, as_json=True)
        return
    replay = " (already present)" if result["idempotent_replay"] else ""
    print(
        f"Published performance-task release {result['release_id']}{replay}: "
        f"{result['task_count']} task(s), pinned to "
        f"{result['corpus_release_id']}."
    )
    print("All resulting evidence remains shadow-only until a later reviewed model boundary.")


def command_task_releases(args: argparse.Namespace) -> None:
    rows = PerformanceLedger(_inspection_database(args)).list_releases()
    if args.json:
        _emit(rows, as_json=True)
        return
    if not rows:
        print("No performance-task releases are installed.")
        return
    for row in rows:
        print(
            f"{row['id']}  tasks={row['task_count']} "
            f"approved={row['approved_count'] or 0} "
            f"pilot={row['pilot_count'] or 0} "
            f"quarantined={row['quarantined_count'] or 0}"
        )


def command_task_list(args: argparse.Namespace) -> None:
    rows = PerformanceLedger(_inspection_database(args)).list_tasks(
        release_id=args.release,
        status=args.status,
    )
    if args.json:
        _emit(rows, as_json=True)
        return
    if not rows:
        print("No performance tasks matched.")
        return
    for row in rows:
        print(
            f"{row['task_id']}@{row['task_version']} [{row['status']}] "
            f"{row['modality']}: {row['title']} ({row['release_id']})"
        )


def command_task_show(args: argparse.Namespace) -> None:
    result = PerformanceLedger(_inspection_database(args)).show_task(
        args.task,
        task_version=args.version,
        release_id=args.release,
    )
    if args.json:
        _emit(result, as_json=True)
        return
    task = result["task"]
    print(
        f"{task['id']}@{task['version']} [{result['status']}] "
        f"{task['modality']}: {task['title']}"
    )
    print(f"  release: {result['release_id']} -> {result['corpus_release_id']}")
    print(f"  family: {task['family_id']}")
    print(f"  instructions: {task['instructions']}")
    print("  rubric:")
    for criterion in task["criteria"]:
        concepts = ", ".join(
            concept_id for concept_id, _ in criterion["concept_weights"]
        )
        objectives = ", ".join(
            objective_id for objective_id, _ in criterion["objective_weights"]
        )
        binding = f"concepts={concepts or 'none'}"
        if objectives:
            binding += f"; objectives={objectives}"
        print(f"    {criterion['id']}: {criterion['name']} ({binding})")
    print("  semantic checkpoints:")
    for action_type in task["allowed_action_kinds"]:
        contract = ACTION_PAYLOAD_CONTRACTS[ActionKind(action_type)]
        fields = ", ".join(
            f"{name}:{value_type}" for name, value_type in contract.items()
        )
        print(f"    {action_type}: {{{fields}}}")
    if task["allowed_tool_ids"] is None:
        print("  tools: no task-specific allowlist")
    else:
        print(f"  tools: {', '.join(task['allowed_tool_ids']) or 'none'}")
    print(f"  task digest: {result['task_digest']}")
    print("  evidence boundary: shadow only; no mastery or certification update")


def command_task_recommend(args: argparse.Namespace) -> None:
    result = recommend_performance_tasks(
        _inspection_database(args),
        args.session,
        limit=args.limit,
    )
    if args.json:
        _emit(result, as_json=True)
        return
    if not result["recommendations"]:
        print("No released productive-skill probe matches this session scope.")
        return
    print(
        "Optional productive-skill probes (read-only recommendation; "
        "shadow evidence only):"
    )
    for rank, item in enumerate(result["recommendations"], start=1):
        reasons = ", ".join(reason.replace("_", " ") for reason in item["reasons"])
        print(
            f"  {rank}. {item['task_id']}@{item['task_version']} "
            f"[{item['status']}/{item['modality']}] score={item['score']:.3f}"
        )
        print(f"     release {item['task_release_id']} · {reasons}")
    boundary = result["selection_boundary"]
    if not boundary["startable_now"]:
        blockers = ", ".join(
            blocker["code"].replace("_", " ")
            for blocker in boundary["start_blockers"]
        )
        print(f"Start currently blocked by: {blockers}.")
    print("Nothing was started; use `tsq task start` with the exact task and release.")


def command_task_start(args: argparse.Namespace) -> None:
    result = PerformanceLedger(_database(args)).start_attempt(
        args.session,
        args.task,
        task_version=args.version,
        task_release_id=args.release,
        idempotency_key=args.idempotency_key,
    )
    if args.json:
        _emit(result, as_json=True)
        return
    print(
        f"Started shadow performance attempt {result['id']} for "
        f"{result['task_id']}@{result['task_version']}."
    )
    print(result["task"]["instructions"])


def command_task_action(args: argparse.Namespace) -> None:
    result = PerformanceLedger(_database(args)).record_action(
        args.attempt,
        args.action_type,
        _action_payload(args),
        phase=args.phase,
        idempotency_key=args.idempotency_key,
    )
    if args.json:
        _emit(result, as_json=True)
        return
    replay = " (idempotent replay)" if result["idempotent_replay"] else ""
    print(
        f"Recorded {result['action_type']} sequence {result['sequence']} "
        f"for {result['attempt_id']}{replay}."
    )


def command_task_actions(args: argparse.Namespace) -> None:
    result = PerformanceLedger(_inspection_database(args)).list_actions(args.attempt)
    if args.json:
        _emit(result, as_json=True)
        return
    for action in result:
        print(
            f"{action['sequence']:>3}  {action['phase']:<13} "
            f"{action['action_type']:<24} {action['elapsed_ms']} ms"
        )


def command_task_claims(args: argparse.Namespace) -> None:
    result = PerformanceLedger(_inspection_database(args)).list_scoring_claims(
        attempt_id=args.attempt,
        status=args.status,
    )
    if args.json:
        _emit(result, as_json=True)
        return
    if not result:
        print("No scoring callback admissions matched.")
        return
    for claim in result:
        print(
            f"{claim['status']:<10} {claim['id']}  "
            f"{claim['provider_id']}@{claim['provider_version']}  "
            f"attempt={claim['attempt_id']}"
        )
    if any(claim["status"] == "unresolved" for claim in result):
        print(
            "Unresolved admissions remain fail-closed; TSQ will not retry the "
            "provider automatically."
        )


def command_task_score(args: argparse.Namespace) -> None:
    if args.provider != "deterministic-test":
        raise ValidationError(f"Unsupported performance provider: {args.provider!r}.")
    for option, value in (
        ("--score", args.score),
        ("--reliability", args.reliability),
    ):
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValidationError(
                f"{option} must be finite and between 0 and 1."
            )
    ledger = PerformanceLedger(_database(args))
    report = ledger.report(args.attempt)
    submitted = [
        action for action in report["actions"] if action["action_type"] == "submitted"
    ]
    if len(submitted) != 1:
        raise ValidationError(
            "The deterministic test provider requires exactly one submitted checkpoint."
        )
    imported = ImportedEvaluation(
        criteria=tuple(
            ImportedCriterionResult(
                criterion_id=criterion["id"],
                status=EvaluationStatus.VALID,
                score=args.score,
                outcome_code="synthetic_fixture",
                phase=ActionPhase.UNASSISTED,
                source_action_ids=(submitted[0]["id"],),
                reliability=args.reliability,
            )
            for criterion in report["task"]["criteria"]
        )
    )
    provider = SyntheticDeterministicProvider(imported)
    registry = ScoringProviderRegistry(allow_synthetic=True)
    registry.register(provider, provider.authority_binding)
    result = ledger.score_attempt(
        args.attempt,
        registry,
        provider.provider_id,
        provider.provider_version,
        idempotency_key=args.idempotency_key,
    )
    if args.json:
        _emit(result, as_json=True)
        return
    print(
        f"Synthetic score recorded for {args.attempt}; candidate shadow weight "
        f"{result['shadow_evidence']['total_evidence_weight']:.3f}."
    )
    print("Mastery applied: no. Certification applied: no.")


def command_task_import_evaluation(args: argparse.Namespace) -> None:
    try:
        size = args.path.stat().st_size
    except OSError as exc:
        raise ValidationError(f"Could not inspect evaluation {args.path}: {exc}") from exc
    if size > 1024 * 1024:
        raise ValidationError("Imported evaluation exceeds the 1 MiB limit.")
    try:
        raw = args.path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"Could not read evaluation {args.path}: {exc}") from exc
    if len(raw.encode("utf-8")) > 1024 * 1024:
        raise ValidationError("Imported evaluation exceeds the 1 MiB limit.")
    try:
        imported = ImportedEvaluation.from_json(raw)
    except ValueError as exc:
        raise ValidationError(f"Imported evaluation is invalid: {exc}") from exc
    result = PerformanceLedger(_database(args)).import_evaluation(
        args.attempt,
        imported,
        provider_id=args.provider_id,
        provider_version=args.provider_version,
        declared_kind=ScorerKind(args.declared_kind),
        idempotency_key=args.idempotency_key,
    )
    if args.json:
        _emit(result, as_json=True)
        return
    print(f"Imported evaluation {result['evaluation']['id']} as shadow evidence.")
    print("Mastery applied: no. Certification applied: no.")


def command_task_report(args: argparse.Namespace) -> None:
    result = PerformanceLedger(_inspection_database(args)).report(args.attempt)
    if args.json:
        _emit(result, as_json=True)
        return
    print(
        f"Attempt {result['id']} [{result['status']}] "
        f"{result['task_id']}@{result['task_version']}"
    )
    print(
        f"  actions: {result['action_count']}  hints: {result['hint_count']}  "
        f"checks: {result['check_count']}  elapsed: {result['elapsed_ms']} ms"
    )
    print(f"  evaluations: {result['evaluation_count']} (all shadow ledger entries)")
    for item in result["evaluations"]:
        shadow = item["shadow_evidence"]
        normalized = item["authority"]["normalized_result"]
        provider = normalized["provider"]
        reported = shadow["reported_task_score"]
        score = "unavailable" if reported is None else f"{reported * 100:.1f}%"
        print(
            f"    {item['evaluation']['id']}: raw task score {score} · "
            f"{provider['provider_id']}@{provider['provider_version']} · "
            f"candidate weight {shadow['total_evidence_weight']:.3f}"
        )
    print("  mastery claim: no  certification claim: no")


def command_replay(args: argparse.Namespace) -> None:
    replayer = ProjectionReplay(Database(args.db))
    if args.check:
        result = replay_or_error(lambda: replayer.check(args.learner))
    else:
        result = replay_or_error(
            lambda: replayer.rebuild_copy(args.learner, args.rebuild_copy)
        )
    if args.json:
        _emit(result, as_json=True)
    elif result["ok"]:
        print(
            f"Projection replay verified {result['response_count']} response "
            f"checkpoint(s) for {result['learner_id']}."
        )
        if result["mode"] == "rebuild-copy":
            print(
                f"Rebuilt copy written to {result['rebuilt_database']}; "
                "the source database was not modified."
            )
    else:
        print("Projection replay failed:")
        for error in result["errors"]:
            print(f"  - {error}")
    if not result["ok"]:
        raise TSQError("Projection replay found inconsistencies.")


def command_verify(args: argparse.Namespace) -> None:
    database = _inspection_database(args, require_corpus=False)
    report = database.verify_integrity(args.stream)
    if args.json:
        _emit(report, as_json=True)
    elif report["ok"]:
        print(
            f"Ledger verified: {report['event_count']} events across "
            f"{report['stream_count']} streams; relational integrity is clean."
        )
    else:
        print("Ledger verification failed:")
        for error in report["errors"]:
            print(f"  - {error}")
    if not report["ok"]:
        raise TSQError("Integrity verification found errors.")


_PEDAGOGICAL_ROLE_LABELS = {
    "main": "PRACTICE",
    "exploration_probe": "EXPLORE",
    "remediation_probe": "TARGETED PRACTICE",
    "verification": "TRANSFER CHECK",
}


def _pedagogical_role_label(role: str) -> str:
    """Translate a stable internal role into a learner-facing cue."""

    return _PEDAGOGICAL_ROLE_LABELS.get(
        role,
        role.replace("_", " ").upper(),
    )


def _print_compact_study_completion(report: dict[str, Any]) -> None:
    """Render a useful study summary without exposing model internals."""

    topic = report["topic"]
    target = topic["name"] if topic else report["root_concept_id"]
    print(
        f"Session {report['session_id']} · {target} · {report['status']}"
    )
    print(
        f"  {report['questions_answered']} completed · "
        f"{report['correct']} correct · "
        f"{report['selected_incorrect']} incorrect · "
        f"{report['abstained']} skipped"
    )
    if report["selected_answers"]:
        print(
            "  Among selected answers: "
            f"{report['correct']}/{report['selected_answers']} correct "
            f"({report['selected_accuracy'] * 100:.1f}%)"
        )
    if report["remediation_questions"]:
        print(
            f"  {report['remediation_questions']} targeted practice or "
            "transfer check question(s)"
        )
    objective_performance = report["objective_performance"]
    if objective_performance:
        print("  Session evidence:")
        for objective in objective_performance[:4]:
            observed = objective["session"]
            selected_evidence = (
                f"{observed['correct']}/{observed['selected_answers']} "
                "selected answers correct"
                if observed["selected_answers"]
                else "0 selected answers"
            )
            skipped = (
                f" · {observed['abstained']} skipped"
                if observed["abstained"]
                else ""
            )
            print(f"    {objective['name']}: {selected_evidence}{skipped}")
        if len(objective_performance) > 4:
            print(
                f"    +{len(objective_performance) - 4} additional "
                "objective(s)"
            )
    print(
        "  These are provisional session signals, not a final skill rating; "
        "independent and delayed checks are still needed."
    )
    print(
        "  For full evidence on future sessions, add --details. This session "
        f"remains available as {report['session_id']} in the same database."
    )


def command_study(args: argparse.Namespace) -> None:
    if args.limit < 1:
        raise ValidationError("Study limit must be a positive integer.")
    database = _database(args)
    engine = AdaptiveEngine(database)
    engine.create_learner(args.learner, args.name)
    session = engine.start_session(args.learner, args.topic, mode=args.mode, seed=args.seed)
    topic_name = args.topic
    if session.get("topic_id"):
        topic_name = database.resolve_topic(
            session["topic_id"], session["corpus_release_id"]
        )["name"]
    print(f"Session {session['id']} · {topic_name} · {args.mode}")
    print("Use 1-4 to answer, ? for 'I do not know', or q to stop.\n")

    completed = 0
    while completed < args.limit:
        try:
            presentation = engine.next_question(session["id"])
        except ExhaustedError as exc:
            engine.end_session(
                session["id"],
                status="completed",
                reason="safe_topic_scope_exhausted",
            )
            print(f"Session complete: {exc}\n")
            break
        role_label = _pedagogical_role_label(presentation.pedagogical_role)
        print(
            f"[{role_label}] "
            f"{presentation.question.kind.value.replace('_', ' ')}"
        )
        if presentation.question.objective is not None:
            objective = presentation.question.objective
            print(f"Focus: {objective.name}")
            if args.explain_policy:
                print(
                    f"  objective internals: id={objective.id}; "
                    f"operation={objective.operation.value}; "
                    f"evidence_type={objective.evidence_type}"
                )
        print(presentation.question.stem)
        ordered = presentation.ordered_options
        for index, option in enumerate(ordered, start=1):
            print(f"  {index}. {option.text}")
        if args.explain_policy:
            print(
                f"  policy: phase={presentation.phase.value}; "
                f"pedagogical_role={presentation.pedagogical_role}; "
                f"{presentation.rationale}"
            )
        started = time.perf_counter()
        while True:
            try:
                raw = input("answer> ").strip().lower()
            except EOFError:
                engine.end_session(
                    session["id"], status="abandoned", reason="input_closed"
                )
                print("\nSession stopped; all completed evidence is saved.")
                return
            except KeyboardInterrupt:
                engine.end_session(
                    session["id"], status="abandoned", reason="interrupted"
                )
                raise
            if raw == "q":
                engine.end_session(
                    session["id"], status="abandoned", reason="user_quit"
                )
                print("Session stopped; all completed evidence is saved.")
                return
            if raw == "?":
                selected_id = None
                break
            if raw.isdigit() and 1 <= int(raw) <= len(ordered):
                selected_id = ordered[int(raw) - 1].id
                break
            print("Enter 1-4, ?, or q.")
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        confidence = None
        if args.ask_confidence and selected_id is not None:
            while True:
                try:
                    raw_confidence = input(
                        "confidence 0-100 (blank to skip)> "
                    ).strip()
                except EOFError:
                    engine.end_session(
                        session["id"], status="abandoned", reason="input_closed"
                    )
                    print("\nSession stopped; all completed evidence is saved.")
                    return
                except KeyboardInterrupt:
                    engine.end_session(
                        session["id"], status="abandoned", reason="interrupted"
                    )
                    raise
                if not raw_confidence:
                    break
                try:
                    number = float(raw_confidence)
                except ValueError:
                    print("Enter a number from 0 to 100.")
                    continue
                if 0 <= number <= 100:
                    confidence = number / 100.0
                    break
                print("Enter a number from 0 to 100.")
        result = engine.submit_answer(
            presentation.decision_id,
            selected_id,
            confidence=confidence,
            response_ms=elapsed_ms,
            feedback_shown=False,
            idempotency_key=new_id("cli"),
        )
        if selected_id is None:
            print("\n— Skipped — you chose 'I do not know'")
        else:
            print("\n✓ Correct" if result.correct else "\n✗ Not correct")
        if result.selected_option and not result.correct:
            print(f"Why that choice fails: {result.selected_option.rationale}")
        print(f"Best answer: {result.correct_option.text}")
        print(f"Why: {result.correct_option.rationale}")
        if result.focus_misconception_id:
            if args.explain_policy:
                print(
                    "Next probe targets hypothesis: "
                    f"{result.focus_misconception_id}"
                )
            else:
                print(
                    "Next practice will check the interpretation behind "
                    "that choice."
                )
        objective_names = {
            objective.id: objective.name
            for objective in database.get_learning_objectives(
                session["corpus_release_id"]
            )
        }
        assessed_change = next(
            (
                change
                for change in result.state_changes
                if "objective_id" in change
            ),
            None,
        )
        if assessed_change is not None:
            objective_id = assessed_change["objective_id"]
            objective_name = objective_names.get(objective_id, objective_id)
            if args.explain_policy:
                print(
                    "Objective projection: "
                    f"{objective_name} ({objective_id}) · "
                    f"{assessed_change['prior_mastery'] * 100:.1f}% → "
                    f"{assessed_change['posterior_mastery'] * 100:.1f}%"
                )
            else:
                print(f"Evidence recorded for: {objective_name}")
        if result.focus_objective_id:
            focus_name = objective_names.get(
                result.focus_objective_id,
                result.focus_objective_id,
            )
            if args.explain_policy:
                print(
                    f"Next probe objective: {focus_name} "
                    f"({result.focus_objective_id})"
                )
            else:
                print(f"Next focus: {focus_name}")
        if result.transition_reason == "descend_to_evidence_boundary":
            focus_name = (
                database.get_graph(session["corpus_release_id"])
                .concepts[result.focus_concept_id]
                .name
                if result.focus_concept_id
                else "a prerequisite"
            )
            print(
                "The next probe steps down to the strongest evidence boundary: "
                f"{focus_name}."
            )
        _record_cli_feedback(
            engine,
            decision_id=presentation.decision_id,
            selected_option_id=(
                result.selected_option.id if result.selected_option else None
            ),
            correct_option_id=result.correct_option.id,
            selected_rationale=(
                result.selected_option.rationale
                if result.selected_option
                else None
            ),
            correct_rationale=result.correct_option.rationale,
        )
        print()
        completed += 1

    current = database.get_session(session["id"])
    if current["status"] == "active":
        engine.end_session(
            session["id"],
            status="completed",
            reason="question_limit_reached",
        )
    if args.details:
        print(f"Completed {completed} questions.\n")
        command_session_report(
            argparse.Namespace(db=args.db, session=session["id"], json=False)
        )
        print()
        command_profile(
            argparse.Namespace(
                db=args.db,
                learner=args.learner,
                topic=args.topic,
                json=False,
            )
        )
    else:
        _print_compact_study_completion(engine.session_report(session["id"]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tsq",
        description="Explainable, knowledge-graph adaptive learning engine",
    )
    parser.add_argument("--db", type=Path, default=_default_database(), help="SQLite database path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser(
        "start", help="Initialize if needed and begin an interactive study session"
    )
    start.add_argument("--learner", default="me")
    start.add_argument(
        "--name",
        help="Display name for a new learner; omit for an existing learner",
    )
    start.add_argument(
        "--topic",
        help="Topic ID or friendly topic name (interactive choice when omitted)",
    )
    start.add_argument(
        "--mode", choices=["learn", "diagnose", "review"], default="learn"
    )
    start.add_argument("--limit", type=int, default=5)
    start.add_argument("--seed", type=int)
    start_confidence = start.add_mutually_exclusive_group()
    start_confidence.add_argument(
        "--ask-confidence",
        dest="ask_confidence",
        action="store_true",
        default=True,
        help="collect confidence after each answer (the default)",
    )
    start_confidence.add_argument(
        "--no-confidence",
        dest="ask_confidence",
        action="store_false",
        help="skip confidence collection; answers remain lower-certainty evidence",
    )
    start.add_argument("--explain-policy", action="store_true")
    start.add_argument(
        "--details",
        action="store_true",
        help="show the full session and learner evidence reports on completion",
    )
    start.set_defaults(func=command_start)

    init = subparsers.add_parser("init", help="Initialize a database and import a corpus")
    init.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help=f"Corpus JSON path (default: bundled {BUNDLED_CORPUS_NAME})",
    )
    init.add_argument("--json", action="store_true")
    init.set_defaults(func=command_init)

    importer = subparsers.add_parser("import", help="Validate and import a versioned corpus bundle")
    importer.add_argument("path", type=Path)
    importer.add_argument("--json", action="store_true")
    importer.set_defaults(func=command_import)

    revoke = subparsers.add_parser(
        "revoke",
        help="Emergency-quarantine a question across all pinned releases",
    )
    revoke.add_argument("question")
    revoke.add_argument("--reason", required=True)
    revoke.add_argument("--idempotency-key")
    revoke.add_argument("--json", action="store_true")
    revoke.set_defaults(func=command_revoke)

    audit = subparsers.add_parser("audit", help="Run deterministic corpus-quality gates")
    audit.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=None,
        help=f"Corpus JSON path (default: bundled {BUNDLED_CORPUS_NAME})",
    )
    audit.add_argument("--json", action="store_true")
    audit.add_argument("--strict", action="store_true", help="Treat warnings as a failing audit")
    audit.set_defaults(func=command_audit)

    capacity = subparsers.add_parser(
        "capacity",
        help="Measure exact sustained main-family capacity and safety reserves",
    )
    capacity.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=None,
        help=f"Corpus JSON path (default: bundled {BUNDLED_CORPUS_NAME})",
    )
    capacity_scope = capacity.add_mutually_exclusive_group()
    capacity_scope.add_argument("--concept", help="Analyze one stable concept ID")
    capacity_scope.add_argument("--topic", help="Analyze one stable topic ID")
    capacity_scope.add_argument(
        "--all",
        action="store_true",
        help="Analyze every curriculum topic (the default)",
    )
    capacity.add_argument("--strict", action="store_true")
    capacity.add_argument(
        "--target-main-count",
        type=int,
        help=(
            "Require this many safely serviceable credible-correct main "
            "families per selected target instead of the concept-count default"
        ),
    )
    capacity.add_argument(
        "--state-limit",
        type=int,
        default=DEFAULT_STATE_LIMIT,
        help="Maximum exact search states per target (never falls back to a heuristic)",
    )
    capacity.add_argument("--json", action="store_true")
    capacity.set_defaults(func=command_capacity)

    topics = subparsers.add_parser("topics", help="List the curriculum topic hierarchy")
    topics.add_argument(
        "--concepts",
        action="store_true",
        help="Show low-level assessable objectives instead of curriculum topics",
    )
    topics.add_argument("--json", action="store_true")
    topics.set_defaults(func=command_topics)

    graph = subparsers.add_parser("graph", help="Inspect a topic's learning scope")
    graph.add_argument("topic")
    graph.add_argument("--json", action="store_true")
    graph.set_defaults(func=command_graph)

    learner = subparsers.add_parser("learner", help="Manage learners")
    learner_sub = learner.add_subparsers(dest="learner_command", required=True)
    learner_add = learner_sub.add_parser("add")
    learner_add.add_argument("learner_id")
    learner_add.add_argument(
        "--name", help="Display name (defaults to the learner ID)"
    )
    learner_add.add_argument("--json", action="store_true")
    learner_add.set_defaults(func=command_learner_add)

    session = subparsers.add_parser(
        "session", help="Create, inspect, or close adaptive sessions"
    )
    session_sub = session.add_subparsers(dest="session_command", required=True)
    session_list = session_sub.add_parser(
        "list", help="List recoverable session history"
    )
    session_list.add_argument("--learner", help="Filter by exact learner ID")
    session_list.add_argument(
        "--status", choices=["active", "completed", "abandoned"]
    )
    session_list.add_argument("--limit", type=int, default=20)
    session_list.add_argument("--json", action="store_true")
    session_list.set_defaults(func=command_session_list)

    session_start = session_sub.add_parser("start")
    session_start.add_argument("--learner", required=True)
    session_start.add_argument(
        "--name",
        help="Display name for a new learner; omit for an existing learner",
    )
    session_start.add_argument("--topic", required=True)
    session_start.add_argument("--mode", choices=["learn", "diagnose", "review"], default="learn")
    session_start.add_argument("--seed", type=int)
    session_start.add_argument("--idempotency-key")
    session_start.add_argument("--json", action="store_true")
    session_start.set_defaults(func=command_session_start)

    session_end = session_sub.add_parser("end")
    session_end.add_argument("session")
    session_end.add_argument(
        "--status", choices=["completed", "abandoned"], default="completed"
    )
    session_end.add_argument("--reason")
    session_end.add_argument("--idempotency-key")
    session_end.add_argument("--json", action="store_true")
    session_end.set_defaults(func=command_session_end)

    session_report = session_sub.add_parser(
        "report", help="Show timing, difficulty, continuity, and evidence metrics"
    )
    session_report.add_argument("session")
    session_report.add_argument("--json", action="store_true")
    session_report.set_defaults(func=command_session_report)

    next_parser = subparsers.add_parser("next", help="Select or retrieve the pending question")
    next_parser.add_argument("session")
    next_parser.add_argument("--explain", action="store_true")
    next_parser.add_argument("--json", action="store_true")
    next_parser.set_defaults(func=command_next)

    answer = subparsers.add_parser("answer", help="Submit one immutable response event")
    answer.add_argument("decision")
    answer.add_argument("option", help="Stable option ID, or ? for I do not know")
    answer.add_argument(
        "--confidence",
        type=float,
        help=(
            "Confidence from 0 to 1; ignored for a new abstention because no "
            "option was selected"
        ),
    )
    answer.add_argument("--response-ms", type=int)
    answer.add_argument("--hints", type=int, default=0)
    answer.add_argument("--idempotency-key")
    answer.add_argument("--json", action="store_true")
    answer.set_defaults(func=command_answer)

    action = subparsers.add_parser(
        "action",
        help="Record or inspect privacy-minimized semantic learner actions",
    )
    action_sub = action.add_subparsers(dest="action_command", required=True)
    action_record = action_sub.add_parser(
        "record", help="Append one immutable observational action"
    )
    action_record.add_argument("decision")
    action_record.add_argument(
        "action_type", choices=[kind.value for kind in ActionKind]
    )
    action_payload = action_record.add_mutually_exclusive_group(required=True)
    action_payload.add_argument(
        "--payload", help="Exact action payload as a JSON object"
    )
    action_payload.add_argument(
        "--payload-file", type=Path, help="Read the exact JSON payload from a file"
    )
    action_record.add_argument(
        "--stage",
        choices=["unassisted", "assisted", "post_feedback"],
        default="unassisted",
    )
    action_record.add_argument("--artifact-sha256")
    action_record.add_argument("--artifact-size-bytes", type=int)
    action_record.add_argument("--artifact-media-type")
    action_record.add_argument("--idempotency-key")
    action_record.add_argument("--json", action="store_true")
    action_record.set_defaults(func=command_action_record)

    action_list = action_sub.add_parser(
        "list", help="List the semantic trace for one decision"
    )
    action_list.add_argument("decision")
    action_list.add_argument("--json", action="store_true")
    action_list.set_defaults(func=command_action_list)

    action_kinds = action_sub.add_parser(
        "kinds", help="Show the exact allowlisted payload contracts"
    )
    action_kinds.add_argument("--json", action="store_true")
    action_kinds.set_defaults(func=command_action_kinds)

    study = subparsers.add_parser("study", help="Run an interactive adaptive CLI session")
    study.add_argument("--learner", required=True)
    study.add_argument(
        "--name",
        help="Display name for a new learner; omit for an existing learner",
    )
    study.add_argument("--topic", required=True)
    study.add_argument("--mode", choices=["learn", "diagnose", "review"], default="learn")
    study.add_argument("--limit", type=int, default=10)
    study.add_argument("--seed", type=int)
    study_confidence = study.add_mutually_exclusive_group()
    study_confidence.add_argument(
        "--ask-confidence",
        dest="ask_confidence",
        action="store_true",
        default=True,
        help="collect confidence after each answer (the default)",
    )
    study_confidence.add_argument(
        "--no-confidence",
        dest="ask_confidence",
        action="store_false",
        help="skip confidence collection; answers remain lower-certainty evidence",
    )
    study.add_argument("--explain-policy", action="store_true")
    study.add_argument(
        "--details",
        action="store_true",
        help="show the full session and learner evidence reports on completion",
    )
    study.set_defaults(func=command_study)

    profile = subparsers.add_parser("profile", help="Show the probabilistic learner projection")
    profile.add_argument("--learner", required=True)
    profile.add_argument("--topic")
    profile.add_argument("--json", action="store_true")
    profile.set_defaults(func=command_profile)

    trace = subparsers.add_parser("trace", help="Explain every adaptive decision in a session")
    trace.add_argument("session")
    trace.add_argument("--json", action="store_true")
    trace.set_defaults(func=command_trace)

    coverage = subparsers.add_parser(
        "coverage", help="Plan corpus growth from explicit concept/kind coverage debt"
    )
    coverage.add_argument("--limit", type=int, default=25)
    coverage_scope = coverage.add_mutually_exclusive_group()
    coverage_scope.add_argument(
        "--topic",
        help="restrict to a curriculum topic ID or exact topic name",
    )
    coverage_scope.add_argument(
        "--concept",
        help="restrict to one stable concept ID",
    )
    coverage.add_argument(
        "--objective",
        help="restrict to one stable learning-objective ID",
    )
    coverage.add_argument(
        "--misconception",
        help="restrict to an exact targeted misconception ID",
    )
    coverage.add_argument(
        "--kind",
        choices=sorted(CoveragePlanner.KIND_TARGETS),
        help="restrict to one planned question kind",
    )
    coverage.add_argument(
        "--goal",
        choices=(
            "concept_kind",
            "objective_misconception_serviceability",
            "objective_serviceability",
        ),
        help="restrict to one structural authoring goal",
    )
    coverage.add_argument(
        "--maximum-difficulty",
        type=float,
        help="restrict to authored-prior difficulty at or below this value",
    )
    coverage.add_argument("--enqueue", action="store_true")
    coverage.add_argument("--json", action="store_true")
    coverage.set_defaults(func=command_coverage)

    jobs = subparsers.add_parser(
        "jobs", help="Operate quarantined offline generation jobs"
    )
    jobs_sub = jobs.add_subparsers(dest="jobs_command", required=True)

    jobs_list = jobs_sub.add_parser("list", help="List generation jobs")
    jobs_list.add_argument(
        "--status", choices=sorted(AuthoringJobs.STATUSES), help="Filter by job status"
    )
    jobs_list.add_argument("--limit", type=int, default=50)
    jobs_list.add_argument("--json", action="store_true")
    jobs_list.set_defaults(func=command_jobs_list)

    jobs_show = jobs_sub.add_parser("show", help="Inspect one job and all attempts")
    jobs_show.add_argument("job")
    jobs_show.add_argument("--json", action="store_true")
    jobs_show.set_defaults(func=command_jobs_show)

    jobs_run = jobs_sub.add_parser(
        "run", help="Run a planned job with an explicit offline provider"
    )
    jobs_run.add_argument("job")
    jobs_run.add_argument(
        "--provider",
        required=True,
        choices=["deterministic-test"],
        help="Provider adapter (only the explicit test fixture is bundled)",
    )
    jobs_run.add_argument(
        "--source-context",
        type=Path,
        required=True,
        help=(
            "UTF-8 approved source file; the file is not stored directly, "
            "while generated/reviewer artifacts are retained with exact-context "
            "redaction"
        ),
    )
    jobs_run.add_argument("--json", action="store_true")
    jobs_run.set_defaults(func=command_jobs_run)

    jobs_retry = jobs_sub.add_parser(
        "retry", help="Return a rejected or failed job to planned state"
    )
    jobs_retry.add_argument("job")
    jobs_retry.add_argument(
        "--recover-running",
        action="store_true",
        help="Fail a stranded running attempt before retrying it",
    )
    jobs_retry.add_argument("--json", action="store_true")
    jobs_retry.set_defaults(func=command_jobs_retry)

    reviews = subparsers.add_parser(
        "reviews", help="Inspect independent generation review attestations"
    )
    reviews_sub = reviews.add_subparsers(dest="reviews_command", required=True)
    reviews_show = reviews_sub.add_parser("show", help="Show reviews for one job")
    reviews_show.add_argument("job")
    reviews_show.add_argument("--json", action="store_true")
    reviews_show.set_defaults(func=command_reviews_show)

    quarantine = subparsers.add_parser(
        "quarantine",
        help="Inspect release-pinned questions awaiting human review",
    )
    quarantine_sub = quarantine.add_subparsers(
        dest="quarantine_command",
        required=True,
    )
    quarantine_list = quarantine_sub.add_parser(
        "list",
        help="List quarantined review candidates",
    )
    quarantine_scope = quarantine_list.add_mutually_exclusive_group()
    quarantine_scope.add_argument(
        "--topic",
        help="restrict to a curriculum topic ID or exact topic name",
    )
    quarantine_scope.add_argument(
        "--concept",
        help="restrict to one stable concept ID",
    )
    quarantine_list.add_argument(
        "--objective",
        help="restrict to one stable learning-objective ID",
    )
    quarantine_list.add_argument("--limit", type=int, default=50)
    quarantine_list.add_argument("--json", action="store_true")
    quarantine_list.set_defaults(func=command_quarantine_list)

    quarantine_show = quarantine_sub.add_parser(
        "show",
        help="Show a full quarantined item, rationales, and sources",
    )
    quarantine_show.add_argument("question")
    quarantine_show.add_argument("--json", action="store_true")
    quarantine_show.set_defaults(func=command_quarantine_show)

    quarantine_packet = quarantine_sub.add_parser(
        "packet",
        help="Build a content-bound, non-activating review packet",
    )
    quarantine_packet.add_argument("question")
    quarantine_packet.add_argument(
        "--stage",
        choices=("combined", "blind", "critic"),
        default="combined",
        help=(
            "export coordinator-combined material or one isolated review stage"
        ),
    )
    quarantine_packet.add_argument("--json", action="store_true")
    quarantine_packet.set_defaults(func=command_quarantine_packet)

    task = subparsers.add_parser(
        "task",
        help="Operate immutable productive-skill tasks and shadow evidence",
    )
    task_sub = task.add_subparsers(dest="task_command", required=True)

    task_import = task_sub.add_parser(
        "import", help="Publish a reviewed task release pinned to a corpus release"
    )
    task_import.add_argument("path", type=Path)
    task_import.add_argument("--json", action="store_true")
    task_import.set_defaults(func=command_task_import)

    task_releases = task_sub.add_parser(
        "releases", help="List installed immutable task releases"
    )
    task_releases.add_argument("--json", action="store_true")
    task_releases.set_defaults(func=command_task_releases)

    task_list = task_sub.add_parser("list", help="List released performance tasks")
    task_list.add_argument("--release")
    task_list.add_argument("--status", choices=sorted({"quarantined", "pilot", "approved"}))
    task_list.add_argument("--json", action="store_true")
    task_list.set_defaults(func=command_task_list)

    task_show = task_sub.add_parser("show", help="Inspect one exact task contract")
    task_show.add_argument("task")
    task_show.add_argument("--version", type=int)
    task_show.add_argument("--release")
    task_show.add_argument("--json", action="store_true")
    task_show.set_defaults(func=command_task_show)

    task_recommend = task_sub.add_parser(
        "recommend",
        help="Rank optional released skill probes from session uncertainty",
    )
    task_recommend.add_argument("--session", required=True)
    task_recommend.add_argument("--limit", type=int, default=5)
    task_recommend.add_argument("--json", action="store_true")
    task_recommend.set_defaults(func=command_task_recommend)

    task_start = task_sub.add_parser(
        "start", help="Start an explicit shadow performance attempt"
    )
    task_start.add_argument("session")
    task_start.add_argument("task")
    task_start.add_argument("--version", type=int)
    task_start.add_argument("--release")
    task_start.add_argument("--idempotency-key")
    task_start.add_argument("--json", action="store_true")
    task_start.set_defaults(func=command_task_start)

    task_action = task_sub.add_parser(
        "action", help="Append a content-free semantic task checkpoint"
    )
    task_action.add_argument("attempt")
    task_action.add_argument(
        "action_type",
        choices=[kind.value for kind in ActionKind if kind is not ActionKind.STARTED],
    )
    task_action_payload = task_action.add_mutually_exclusive_group(required=True)
    task_action_payload.add_argument("--payload", help="Exact action payload JSON object")
    task_action_payload.add_argument("--payload-file", type=Path)
    task_action.add_argument(
        "--phase",
        choices=[phase.value for phase in ActionPhase],
        default=ActionPhase.UNASSISTED.value,
    )
    task_action.add_argument("--idempotency-key")
    task_action.add_argument("--json", action="store_true")
    task_action.set_defaults(func=command_task_action)

    task_actions = task_sub.add_parser(
        "actions", help="List one performance attempt's semantic trace"
    )
    task_actions.add_argument("attempt")
    task_actions.add_argument("--json", action="store_true")
    task_actions.set_defaults(func=command_task_actions)

    task_claims = task_sub.add_parser(
        "claims",
        help="List completed or unresolved scorer callback admissions",
    )
    task_claims.add_argument("--attempt")
    task_claims.add_argument(
        "--status", choices=["completed", "unresolved"]
    )
    task_claims.add_argument("--json", action="store_true")
    task_claims.set_defaults(func=command_task_claims)

    task_score = task_sub.add_parser(
        "score", help="Record a visibly synthetic deterministic test score"
    )
    task_score.add_argument("attempt")
    task_score.add_argument(
        "--provider", choices=["deterministic-test"], required=True
    )
    task_score.add_argument("--score", type=float, required=True)
    task_score.add_argument("--reliability", type=float, default=1.0)
    task_score.add_argument("--idempotency-key")
    task_score.add_argument("--json", action="store_true")
    task_score.set_defaults(func=command_task_score)

    task_evaluation = task_sub.add_parser(
        "import-evaluation",
        help="Import authority-free scorer observations as shadow evidence",
    )
    task_evaluation.add_argument("attempt")
    task_evaluation.add_argument("path", type=Path)
    task_evaluation.add_argument("--provider-id", required=True)
    task_evaluation.add_argument("--provider-version", required=True)
    task_evaluation.add_argument(
        "--declared-kind",
        choices=[kind.value for kind in ScorerKind],
        default=ScorerKind.IMPORTED.value,
    )
    task_evaluation.add_argument("--idempotency-key")
    task_evaluation.add_argument("--json", action="store_true")
    task_evaluation.set_defaults(func=command_task_import_evaluation)

    task_report = task_sub.add_parser(
        "report", help="Inspect timing, actions, and unapplied shadow evaluations"
    )
    task_report.add_argument("attempt")
    task_report.add_argument("--json", action="store_true")
    task_report.set_defaults(func=command_task_report)

    replay = subparsers.add_parser(
        "replay", help="Reconstruct and verify a learner projection on a database copy"
    )
    replay.add_argument("--learner", required=True)
    replay_mode = replay.add_mutually_exclusive_group(required=True)
    replay_mode.add_argument(
        "--check",
        action="store_true",
        help="Replay on a temporary copy and compare every committed checkpoint",
    )
    replay_mode.add_argument(
        "--rebuild-copy",
        type=Path,
        help="Write a verified rebuilt copy without modifying the source database",
    )
    replay.add_argument("--json", action="store_true")
    replay.set_defaults(func=command_replay)

    verify = subparsers.add_parser("verify", help="Verify event hash chains and database integrity")
    verify.add_argument("--stream")
    verify.add_argument("--json", action="store_true")
    verify.set_defaults(func=command_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
        return int(result or 0)
    except (TSQError, KeyboardInterrupt) as exc:
        message = str(exc) if str(exc) else "Interrupted."
        print(f"error: {message}", file=sys.stderr)
        return 2
    except sqlite3.Error as exc:
        message = str(exc) if str(exc) else "SQLite database operation failed."
        print(f"error: Database operation failed: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
