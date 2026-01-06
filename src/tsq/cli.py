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

from .corpus import load_bundle, parse_bundle, read_and_parse, validate_bundle
from .authoring import CoveragePlanner
from .engine import AdaptiveEngine
from .errors import TSQError, ValidationError
from .quality import audit_corpus
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
        counts = database.import_corpus(*read_and_parse(corpus_path))
    if args.json:
        _emit({"database": str(database.path), "corpus": corpus_label, **counts}, as_json=True)
    else:
        print(f"Initialized {database.path}")
        print(
            f"Imported {counts['questions']} questions, {counts['concepts']} concepts, "
            f"and {counts['misconceptions']} misconception hypotheses."
        )


def command_import(args: argparse.Namespace) -> None:
    database = _database(args, require_corpus=False)
    counts = database.import_corpus(*read_and_parse(args.path))
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
            _, _, _, _, questions = parse_bundle(bundle)
        except ValidationError as exc:
            issues.extend(exc.issues)
        else:
            issues = audit_corpus(questions)
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


def command_topics(args: argparse.Namespace) -> None:
    database = _database(args)
    concepts = sorted(
        database.get_graph().concepts.values(), key=lambda concept: (concept.domain, concept.name)
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
    scope = graph.learning_scope(args.topic)
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
        "root": args.topic,
        "concepts": [
            {"id": concept_id, "name": graph.concepts[concept_id].name}
            for concept_id in sorted(scope)
        ],
        "edges": edges,
    }
    if args.json:
        _emit(result, as_json=True)
    else:
        print(f"Learning scope for {args.topic}: {len(scope)} concepts")
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
        completed=args.status == "completed",
        reason=args.reason,
        idempotency_key=args.idempotency_key,
    )
    _emit(session, as_json=args.json)


def _presentation_dict(presentation) -> dict[str, Any]:
    return {
        "decision_id": presentation.decision_id,
        "session_id": presentation.session_id,
        "phase": presentation.phase.value,
        "question_id": presentation.question.id,
        "family_id": presentation.question.family_id,
        "kind": presentation.question.kind.value,
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
            f"{skill['state']:<10}  {skill['name']} "
            f"({skill['independent_families']} independent families, "
            f"{skill['operation_kinds']} operations, {skill['delayed_retrievals']} delayed)"
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
