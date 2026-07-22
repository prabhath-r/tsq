# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from importlib.resources import as_file, files
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
from .authoring import AuthoringJobs, CoveragePlanner, deterministic_test_pipeline
from .engine import AdaptiveEngine
from .evidence import ACTION_PAYLOAD_CONTRACTS, ActionKind
from .errors import ExhaustedError, NotFoundError, TSQError, ValidationError
from .graph import KnowledgeGraph
from .quality import audit_corpus
from .replay import ProjectionReplay, replay_or_error
from .store import Database, new_id


BUNDLED_CORPUS_PACKAGE = "tsq.data"
BUNDLED_CORPUS_NAME = "ai_curriculum.json"
BUNDLED_CORPUS_LABEL = f"{BUNDLED_CORPUS_PACKAGE}:{BUNDLED_CORPUS_NAME}"


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
    if require_corpus:
        with database.read() as connection:
            active = connection.execute(
                "SELECT value FROM meta WHERE key = 'active_corpus_release'"
            ).fetchone()
        if not active:
            raise TSQError("The database has no active corpus. Run `tsq init` first.")
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


def _ensure_starter_corpus(database: Database) -> bool:
    """Install the bundled catalog for a new or legacy catalog-less database."""
    with database.read() as connection:
        active = connection.execute(
            "SELECT value FROM meta WHERE key = 'active_corpus_release'"
        ).fetchone()
        catalog_count = (
            connection.execute(
                """SELECT COUNT(*) AS n FROM release_topics
                   WHERE release_id = ?""",
                (active["value"],),
            ).fetchone()["n"]
            if active
            else 0
        )
    if active and catalog_count:
        return False
    with _corpus_path(None) as (corpus_path, _):
        database.import_corpus(
            *read_and_parse(corpus_path, include_catalog=True)
        )
    return True


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
    installed = _ensure_starter_corpus(database)
    topic = _starter_topic(database, args.topic)
    if installed:
        print("Installed the bundled reviewed curriculum catalog.")
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
            targets = (concept_target(args.concept),)
        elif args.topic:
            targets = (topic_target(args.topic, topics),)
        else:
            if not topics:
                raise ValueError(
                    "The corpus has no curriculum topics; select --concept explicitly."
                )
            targets = tuple(
                topic_target(topic.id, topics)
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
    database = _database(args)
    catalog = database.get_catalog()
    if catalog["topics"] and not getattr(args, "concepts", False):
        topic_by_id = {topic["id"]: topic for topic in catalog["topics"]}
        children: dict[str | None, list[dict[str, Any]]] = {}
        for topic in catalog["topics"]:
            children.setdefault(topic["parent_id"], []).append(topic)
        for siblings in children.values():
            siblings.sort(key=lambda item: (item["sort_order"], item["name"], item["id"]))

        rows: list[dict[str, Any]] = []

        def append_topic(topic: dict[str, Any], depth: int, path: list[str]) -> tuple[int, int]:
            child_rows = children.get(topic["id"], [])
            descendant_questions = topic["direct_primary_questions"]
            descendant_objectives = len(topic["concepts"])
            row = {
                **topic,
                "depth": depth,
                "path": [*path, topic["name"]],
            }
            rows.append(row)
            for child in child_rows:
                child_questions, child_objectives = append_topic(
                    child, depth + 1, row["path"]
                )
                descendant_questions += child_questions
                descendant_objectives += child_objectives
            row["scope_primary_questions"] = descendant_questions
            row["scope_objectives"] = descendant_objectives
            return descendant_questions, descendant_objectives

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
                print(
                    f"{indent}{row['name']} ({row['id']}) — "
                    f"{row['scope_primary_questions']} questions, "
                    f"{row['scope_objectives']} objectives"
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
    database = _database(args)
    graph = database.get_graph()
    try:
        topic = database.resolve_topic(args.topic)
    except NotFoundError:
        topic = None
    if topic is not None:
        root = topic["id"]
        root_name = topic["name"]
        target_type = "topic"
        scope = database.topic_scope(root, topic["release_id"])
    else:
        root = args.topic
        root_name = graph.concepts[root].name if root in graph.concepts else root
        target_type = "concept"
        scope = graph.learning_scope(root)
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
        "edges": edges,
    }
    if args.json:
        _emit(result, as_json=True)
    else:
        print(f"Learning scope for {root_name} ({root}): {len(scope)} objectives")
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


def command_session_report(args: argparse.Namespace) -> None:
    report = AdaptiveEngine(_database(args)).session_report(args.session)
    if args.json:
        _emit(report, as_json=True)
        return
    topic = report["topic"]
    target = topic["name"] if topic else report["root_concept_id"]
    accuracy = (
        f"{report['accuracy'] * 100:.1f}%" if report["accuracy"] is not None else "n/a"
    )
    response = report["response_time"]
    difficulty = report["difficulty"]
    print(f"Session {report['session_id']} · {target} · {report['status']}")
    print(
        f"  {report['questions_answered']} answered · {report['correct']} correct "
        f"({accuracy}) · {report['abstained']} unsure"
    )
    print(
        f"  active response time {response['active_seconds']:.1f}s · "
        f"wall time {response['wall_seconds']:.1f}s"
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
            f"{report['remediation_questions']} repair/verification question(s)"
        )
    if report["topic_distribution"]:
        rendered = ", ".join(
            f"{row['name']} {row['n']}" for row in report["topic_distribution"]
        )
        print(f"  topic mix: {rendered}")
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
    print(
        "  difficulty values are authored priors and remain uncalibrated until "
        "sufficient response data exists."
    )


def _presentation_dict(presentation) -> dict[str, Any]:
    return {
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


def command_next(args: argparse.Namespace) -> None:
    database = _database(args)
    presentation = AdaptiveEngine(database).next_question(args.session)
    result = _presentation_dict(presentation)
    if args.json:
        _emit(result, as_json=True)
    else:
        print(f"[{result['phase']}] {result['stem']}")
        for index, option in enumerate(result["options"], start=1):
            print(f"  {index}. {option['text']}  (id: {option['id']})")
        if args.explain:
            print(f"Why: {result['selection']['rationale']}")
        print(f"Decision: {result['decision_id']}")


def _submission_dict(result) -> dict[str, Any]:
    return {
        "interaction_id": result.interaction_id,
        "correct": result.correct,
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


def command_answer(args: argparse.Namespace) -> None:
    database = _database(args)
    option_id = None if args.option in {"?", "unknown", "none"} else args.option
    result = AdaptiveEngine(database).submit_answer(
        args.decision,
        option_id,
        confidence=args.confidence,
        response_ms=args.response_ms,
        hint_count=args.hints,
        idempotency_key=args.idempotency_key,
    )
    payload = _submission_dict(result)
    if args.json:
        _emit(payload, as_json=True)
    else:
        print("Correct." if result.correct else "Not correct.")
        if result.selected_option:
            print(f"Your choice: {result.selected_option.rationale}")
        print(f"Answer: {result.correct_option.text}")
        print(result.correct_option.rationale)
        print(f"Next phase: {result.next_phase.value}")
        if result.focus_misconception_id:
            print(f"Current hypothesis: {result.focus_misconception_id}")
        print(f"Adaptive path: {result.transition_reason}")


def _action_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.payload_file is not None:
        try:
            if args.payload_file.stat().st_size > 16_384:
                raise ValidationError(
                    "Action payload files must not exceed 16384 bytes."
                )
            raw = args.payload_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValidationError(
                f"Could not read action payload file {args.payload_file}: {exc}"
            ) from exc
    else:
        raw = args.payload
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > 16_384:
        raise ValidationError("Action payload JSON must not exceed 16384 bytes.")
    try:
        payload = json.loads(raw)
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
    database = _database(args)
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
    database = _database(args)
    profile = AdaptiveEngine(database).profile(args.learner, root_concept_id=args.topic)
    if args.json:
        _emit(profile, as_json=True)
        return
    print(f"Learner: {profile['learner_id']}")
    assessed = [skill for skill in profile["skills"] if skill["evidence_mass"] > 0]
    if not assessed:
        print("No response evidence yet.")
    for skill in sorted(assessed, key=lambda row: (row["mastery"], row["name"])):
        print(
            f"  P(mastered)={skill['mastery'] * 100:5.1f}%  "
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


def command_trace(args: argparse.Namespace) -> None:
    database = _database(args)
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
    database = _database(args)
    planner = CoveragePlanner(database)
    gaps = planner.gaps(limit=args.limit)
    job_ids = planner.enqueue(gaps) if args.enqueue else []
    payload = {
        "gap_count": len(gaps),
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
    print(f"Top {len(gaps)} corpus coverage gaps")
    for gap in gaps:
        blueprint = gap.blueprint
        print(
            f"  {blueprint.concept_id:<32} {blueprint.kind:<16} "
            f"{gap.current_count}/{gap.target_count}  difficulty={blueprint.target_difficulty:+.2f}"
        )
    if job_ids:
        print(f"Enqueued {len(job_ids)} quarantined authoring jobs.")


def command_jobs_list(args: argparse.Namespace) -> None:
    jobs = AuthoringJobs(_database(args)).list(status=args.status, limit=args.limit)
    if args.json:
        _emit(jobs, as_json=True)
        return
    if not jobs:
        print("No generation jobs matched.")
        return
    for job in jobs:
        blueprint = job["blueprint"]
        print(
            f"{job['id']} [{job['status']}] {blueprint['concept_id']} / "
            f"{blueprint['kind']}  attempts={job['run_count']}"
        )


def command_jobs_show(args: argparse.Namespace) -> None:
    job = AuthoringJobs(_database(args)).show(args.job)
    if args.json:
        _emit(job, as_json=True)
        return
    blueprint = job["blueprint"]
    print(f"Job {job['id']} [{job['status']}]")
    print(
        f"  target: {blueprint['concept_id']} / {blueprint['kind']} "
        f"at difficulty {blueprint['target_difficulty']:+.2f}"
    )
    print(f"  sources: {', '.join(blueprint['source_ids'])}")
    print(f"  attempts: {job['run_count']}")
    for run in job["runs"]:
        finished = run["completed_at"] or "in progress"
        print(
            f"    {run['attempt']}: {run['status']} via "
            f"{run['provider']}/{run['model']} ({finished})"
        )
    if job["raw_output"] is not None:
        print(
            f"  artifact: {job['raw_output'].get('id', '<unknown>')} "
            f"[{job['raw_output'].get('status', '<missing>')}]"
        )
        print("  activation: none (reviewed artifacts remain quarantined)")
    if job["validation"] is not None:
        issue_count = len(job["validation"].get("deterministic_issues", ()))
        review_count = len(job["validation"].get("reviews", ()))
        print(f"  validation: {issue_count} issues; {review_count} independent reviews")


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
    result = AuthoringJobs(_database(args)).reviews(args.job)
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
                f"  attempt {attempt['attempt']}: "
                f"{reviewer.get('reviewer_name', '<unknown>')} -> "
                f"{output.get('verdict', '<invalid>')}"
            )


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
    database = _database(args, require_corpus=False)
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
        print(f"[{presentation.phase.value.upper()}] {presentation.question.kind.value.replace('_', ' ')}")
        print(presentation.question.stem)
        ordered = presentation.ordered_options
        for index, option in enumerate(ordered, start=1):
            print(f"  {index}. {option.text}")
        if args.explain_policy:
            print(f"  policy: {presentation.rationale}")
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
        if args.ask_confidence:
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
            idempotency_key=new_id("cli"),
        )
        print("\n✓ Correct" if result.correct else "\n✗ Not correct")
        if result.selected_option and not result.correct:
            print(f"Why that choice fails: {result.selected_option.rationale}")
        print(f"Best answer: {result.correct_option.text}")
        print(f"Why: {result.correct_option.rationale}")
        if result.focus_misconception_id:
            print(f"Next probe targets hypothesis: {result.focus_misconception_id}")
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
        print()
        completed += 1

    current = database.get_session(session["id"])
    if current["status"] == "active":
        engine.end_session(
            session["id"],
            status="completed",
            reason="question_limit_reached",
        )
    print(f"Completed {completed} questions.\n")
    command_session_report(
        argparse.Namespace(db=args.db, session=session["id"], json=False)
    )
    print()
    command_profile(
        argparse.Namespace(db=args.db, learner=args.learner, topic=args.topic, json=False)
    )


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
    start.add_argument("--name")
    start.add_argument(
        "--topic",
        help="Topic ID or friendly topic name (interactive choice when omitted)",
    )
    start.add_argument(
        "--mode", choices=["learn", "diagnose", "review"], default="learn"
    )
    start.add_argument("--limit", type=int, default=5)
    start.add_argument("--seed", type=int)
    start.add_argument("--ask-confidence", action="store_true")
    start.add_argument("--explain-policy", action="store_true")
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
    learner_add.add_argument("--name")
    learner_add.add_argument("--json", action="store_true")
    learner_add.set_defaults(func=command_learner_add)

    session = subparsers.add_parser("session", help="Create an adaptive session")
    session_sub = session.add_subparsers(dest="session_command", required=True)
    session_start = session_sub.add_parser("start")
    session_start.add_argument("--learner", required=True)
    session_start.add_argument("--name")
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
    answer.add_argument("--confidence", type=float)
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
    study.add_argument("--name")
    study.add_argument("--topic", required=True)
    study.add_argument("--mode", choices=["learn", "diagnose", "review"], default="learn")
    study.add_argument("--limit", type=int, default=10)
    study.add_argument("--seed", type=int)
    study.add_argument("--ask-confidence", action="store_true")
    study.add_argument("--explain-policy", action="store_true")
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
        help="UTF-8 file containing approved source context (content is not persisted)",
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


if __name__ == "__main__":
    raise SystemExit(main())
