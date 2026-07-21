#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0

"""Reproducible large-bank benchmark for adaptive candidate retrieval.

The default run builds a synthetic 100,000-question corpus in a temporary
SQLite database, measures the indexed candidate-ID query, full 600-item
hydration, candidate-scoped exposure aggregation, and a complete policy choice.
It is deliberately not part of the unit-test suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sqlite3
import statistics
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tsq.engine import AdaptiveEngine  # noqa: E402
from tsq.store import CANDIDATE_POOL_SQL, Database  # noqa: E402


CONCEPT_ID = "c_benchmark_candidate_retrieval"
LEARNER_ID = "learner_benchmark"
FIXED_TIMESTAMP = "2026-01-01T00:00:00+00:00"

def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _measure(call: Callable[[], Any], *, rounds: int, warmups: int = 1) -> tuple[Any, dict[str, float]]:
    result: Any = None
    for _ in range(warmups):
        result = call()
    samples: list[float] = []
    for _ in range(rounds):
        started = time.perf_counter_ns()
        result = call()
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return result, {
        "min_ms": min(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": _percentile(samples, 0.95),
        "max_ms": max(samples),
    }


def _seed_bank(database: Database, question_count: int, *, batch_size: int = 2_000) -> str:
    release_id = f"release_benchmark_{question_count}"
    with database.transaction() as connection:
        connection.execute(
            """INSERT INTO concepts(id, content_hash, name, description, domain, prior_mastery)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                CONCEPT_ID,
                _sha(CONCEPT_ID),
                "Candidate retrieval benchmark",
                "Synthetic root used only by the repeatable performance benchmark.",
                "benchmark",
                0.25,
            ),
        )
        connection.execute(
            "INSERT INTO corpus_releases(id, bundle_hash, created_at) VALUES (?, ?, ?)",
            (release_id, _sha(release_id), FIXED_TIMESTAMP),
        )
        connection.execute(
            "INSERT INTO release_concepts(release_id, concept_id) VALUES (?, ?)",
            (release_id, CONCEPT_ID),
        )
        for start in range(0, question_count, batch_size):
            stop = min(question_count, start + batch_size)
            questions = []
            mappings = []
            options = []
            for index in range(start, stop):
                question_id = f"q_benchmark_{index:06d}"
                family_id = f"family_benchmark_{index:06d}"
                difficulty = ((index % 81) - 40) / 20.0
                kind = "application" if index % 2 else "conceptual"
                questions.append(
                    (
                        question_id,
                        1,
                        _sha(question_id),
                        family_id,
                        "calibrated",
                        f"Synthetic benchmark question {index}",
                        kind,
                        difficulty,
                        1.1 + ((index % 7) * 0.05),
                        0.25,
                        0.08,
                        '{"generator":"benchmark_candidate_retrieval"}',
                        '["synthetic","benchmark"]',
                        None,
                        FIXED_TIMESTAMP,
                    )
                )
                mappings.append((question_id, CONCEPT_ID, 1.0, "primary"))
                options.extend(
                    (
                        (question_id, "a", "Benchmark option A", 1, "Synthetic correct rationale.", None),
                        (question_id, "b", "Benchmark option B", 0, "Synthetic distractor rationale.", None),
                        (question_id, "c", "Benchmark option C", 0, "Synthetic distractor rationale.", None),
                        (question_id, "d", "Benchmark option D", 0, "Synthetic distractor rationale.", None),
                    )
                )
            connection.executemany(
                """INSERT INTO questions(
                       id, version, content_hash, family_id, status, stem, kind,
                       difficulty, discrimination, guess_rate, slip_rate,
                       provenance_json, tags_json, revision_of, imported_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                questions,
            )
            connection.executemany(
                """INSERT INTO question_concepts(question_id, concept_id, weight, role)
                   VALUES (?, ?, ?, ?)""",
                mappings,
            )
            connection.executemany(
                """INSERT INTO options(
                       question_id, option_id, text, is_correct, rationale, misconception_id
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                options,
            )
        connection.executemany(
            """INSERT INTO release_questions(
                   release_id, question_id, status, evidence_weight
               ) VALUES (?, ?, ?, ?)""",
            (
                (release_id, f"q_benchmark_{index:06d}", "calibrated", 1.0)
                for index in range(question_count)
            ),
        )
        connection.execute(
            "UPDATE corpus_releases SET sealed_at = ? WHERE id = ?",
            (FIXED_TIMESTAMP, release_id),
        )
        connection.execute(
            "INSERT INTO meta(key, value) VALUES ('active_corpus_release', ?)",
            (release_id,),
        )
    with database.connect() as connection:
        connection.execute("ANALYZE")
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return release_id


def _candidate_parameters(release_id: str, candidate_limit: int) -> tuple[Any, ...]:
    return (
        None,
        None,
        0.0,
        release_id,
        LEARNER_ID,
        "approved",
        "calibrated",
        candidate_limit,
    )


def _benchmark(database: Database, release_id: str, args: argparse.Namespace) -> dict[str, Any]:
    parameters = _candidate_parameters(release_id, args.candidate_limit)
    query_connection = database.connect()
    query_connection.execute("CREATE TEMP TABLE requested_scope(id TEXT PRIMARY KEY)")
    query_connection.execute("INSERT INTO requested_scope(id) VALUES (?)", (CONCEPT_ID,))

    def candidate_ids() -> list[sqlite3.Row]:
        return query_connection.execute(CANDIDATE_POOL_SQL, parameters).fetchall()

    id_rows, candidate_query = _measure(candidate_ids, rounds=args.rounds)

    def hydrated_candidates():
        return database.questions_for_scope(
            {CONCEPT_ID},
            learner_id=LEARNER_ID,
            release_id=release_id,
            target_difficulty=0.0,
            limit=args.candidate_limit,
        )

    questions, retrieval = _measure(hydrated_candidates, rounds=args.rounds)
    question_ids = {question.id for question in questions}
    family_ids = {question.family_id for question in questions}
    exposure, exposure_query = _measure(
        lambda: database.get_exposure_summary(
            LEARNER_ID,
            question_ids=question_ids,
            family_ids=family_ids,
        ),
        rounds=args.rounds,
    )

    engine = AdaptiveEngine(database)
    engine.create_learner(LEARNER_ID, "Benchmark learner")
    selection_samples: list[float] = []
    selected_ids: list[str] = []
    for index in range(args.rounds):
        session = engine.start_session(
            LEARNER_ID,
            CONCEPT_ID,
            mode="learn",
            seed=10_000 + index,
        )
        started = time.perf_counter_ns()
        presentation = engine.next_question(session["id"])
        selection_samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
        selected_ids.append(presentation.question.id)
    selection = {
        "min_ms": min(selection_samples),
        "median_ms": statistics.median(selection_samples),
        "p95_ms": _percentile(selection_samples, 0.95),
        "max_ms": max(selection_samples),
    }

    plan_rows = query_connection.execute(
        "EXPLAIN QUERY PLAN " + CANDIDATE_POOL_SQL, parameters
    ).fetchall()
    query_connection.close()
    with database.connect() as connection:
        page_count = connection.execute("PRAGMA page_count").fetchone()[0]
        page_size = connection.execute("PRAGMA page_size").fetchone()[0]

    return {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "sqlite": sqlite3.sqlite_version,
        },
        "question_count": args.questions,
        "candidate_limit": args.candidate_limit,
        "rounds": args.rounds,
        "returned_candidates": len(id_rows),
        "hydrated_candidates": len(questions),
        "candidate_query": candidate_query,
        "candidate_retrieval_and_hydration": retrieval,
        "candidate_scoped_exposure": exposure_query,
        "full_policy_selection": selection,
        "selected_question_ids": selected_ids,
        "exposure_result_sizes": {
            "questions": len(exposure["questions"]),
            "families": len(exposure["families"]),
        },
        "database_size_bytes": page_count * page_size,
        "query_plan": [row[3] for row in plan_rows],
    }


def _print_human(report: dict[str, Any], database_path: Path, seed_seconds: float) -> None:
    print(
        f"Seeded {report['question_count']:,} questions in {seed_seconds:.2f}s "
        f"({report['database_size_bytes'] / (1024 * 1024):.1f} MiB SQLite file)."
    )
    for name in (
        "candidate_query",
        "candidate_retrieval_and_hydration",
        "candidate_scoped_exposure",
        "full_policy_selection",
    ):
        timing = report[name]
        print(
            f"{name:<36} median={timing['median_ms']:8.2f} ms  "
            f"p95={timing['p95_ms']:8.2f} ms  min={timing['min_ms']:8.2f} ms"
        )
    print("Query plan:")
    for detail in report["query_plan"]:
        print(f"  {detail}")
    print(f"Database: {database_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=int, default=100_000)
    parser.add_argument("--candidate-limit", type=int, default=600)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--db",
        type=Path,
        help="Persist the generated database here; the path must not already exist.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.questions < 1:
        raise SystemExit("--questions must be positive")
    if args.candidate_limit < 1:
        raise SystemExit("--candidate-limit must be positive")
    if args.rounds < 1:
        raise SystemExit("--rounds must be positive")
    if args.db is not None and args.db.exists():
        raise SystemExit(f"refusing to overwrite existing database: {args.db}")

    temporary = tempfile.TemporaryDirectory(prefix="tsq-benchmark-") if args.db is None else None
    manager = temporary if temporary is not None else nullcontext()
    with manager:
        database_path = args.db or Path(temporary.name) / "candidate-retrieval.db"
        database = Database(database_path)
        database.initialize()
        started = time.perf_counter()
        release_id = _seed_bank(database, args.questions)
        seed_seconds = time.perf_counter() - started
        report = _benchmark(database, release_id, args)
        report["seed_seconds"] = seed_seconds
        report["database_path"] = str(database_path)
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            _print_human(report, database_path, seed_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
