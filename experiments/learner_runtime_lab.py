#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0

"""Benchmark TSQ selection, updates, and exact-posterior persistence.

The laboratory uses only SQLite databases created beneath one temporary
directory.  A protected database and its SQLite sidecars are fingerprinted
before and after the run but are never opened through TSQ.  Timed mutations
start from byte-equivalent SQLite backups so every observation measures the
same learner state.

Absolute timings remain machine-specific.  The useful outputs are medians,
nearest-rank p95s, cold-versus-history-rich ratios, and cProfile call stacks
that identify where production time is actually spent.
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import hashlib
import json
import math
import platform
import pstats
import sqlite3
import statistics
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import quote


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tsq.corpus import corpus_source_digest, read_and_parse  # noqa: E402
from tsq.engine import AdaptiveEngine  # noqa: E402
from tsq.errors import ExhaustedError  # noqa: E402
from tsq.learner import MODEL_VERSION, LearnerModel  # noqa: E402
from tsq.objective_posterior import (  # noqa: E402
    GRID_SIZE,
    OBJECTIVE_POSTERIOR_ALGORITHM,
)
from tsq.policy import POLICY_VERSION  # noqa: E402
from tsq.store import SCHEMA_VERSION, Database  # noqa: E402


LAB_VERSION = "learner-runtime-lab-v1"
DEFAULT_CORPUS = PROJECT_ROOT / "corpus"
DEFAULT_PROTECTED_DATABASE = PROJECT_ROOT / "tsq.db"
DEFAULT_TOPIC = "t_transformers"
DEFAULT_HISTORY_RESPONSES = 24
DEFAULT_SAMPLES = 15
DEFAULT_WARMUPS = 3
DEFAULT_PROFILE_TOP = 18
MAX_HISTORY_RESPONSES = 80
MAX_SAMPLES = 50
MAX_WARMUPS = 10
MAX_PROFILE_TOP = 40
START = datetime(2102, 2, 3, 9, 0, tzinfo=timezone.utc)


class PerformanceLabError(RuntimeError):
    """Raised when a benchmark precondition or safety invariant fails."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PerformanceLabError(message)


def file_fingerprint(path: Path) -> dict[str, Any]:
    """Return mutation-sensitive metadata without opening a SQLite connection."""

    resolved = path.resolve()
    if not resolved.exists():
        return {"path": str(resolved), "exists": False}
    require(resolved.is_file(), f"Protected path is not a file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    status = resolved.stat()
    return {
        "path": str(resolved),
        "exists": True,
        "sha256": digest.hexdigest(),
        "size_bytes": status.st_size,
        "mtime_ns": status.st_mtime_ns,
        "inode": status.st_ino,
    }


def database_family_fingerprint(path: Path) -> dict[str, dict[str, Any]]:
    """Fingerprint a database and both conventional sidecars as plain files.

    This function intentionally performs no SQLite operation.  Recording absent
    sidecars too means their creation or removal during a run is also detected.
    """

    resolved = path.resolve()
    return {
        "database": file_fingerprint(resolved),
        "wal": file_fingerprint(Path(f"{resolved}-wal")),
        "shm": file_fingerprint(Path(f"{resolved}-shm")),
    }


def corpus_digest(path: Path) -> str:
    return corpus_source_digest(path)


def latency_summary(samples_ns: list[int]) -> dict[str, Any]:
    require(bool(samples_ns), "A latency summary requires at least one sample.")
    ordered = sorted(samples_ns)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    milliseconds = [sample / 1_000_000.0 for sample in samples_ns]
    ordered_ms = sorted(milliseconds)
    return {
        "sample_count": len(samples_ns),
        "median_ms": round(statistics.median(milliseconds), 6),
        "p95_ms": round(ordered_ms[p95_index], 6),
        "mean_ms": round(statistics.fmean(milliseconds), 6),
        "min_ms": round(ordered_ms[0], 6),
        "max_ms": round(ordered_ms[-1], 6),
        "samples_ms": [round(value, 6) for value in milliseconds],
    }


def clone_database(source: Path, destination: Path) -> None:
    """Create a consistent SQLite snapshot; destination must not exist."""

    require(not destination.exists(), f"Clone destination exists: {destination}")
    uri = f"file:{quote(str(source.resolve()))}?mode=ro"
    source_connection = sqlite3.connect(uri, uri=True, timeout=20.0)
    destination_connection = sqlite3.connect(destination, timeout=20.0)
    try:
        source_connection.execute("PRAGMA query_only = ON")
        source_connection.backup(destination_connection)
    finally:
        destination_connection.close()
        source_connection.close()


def remove_database(path: Path) -> None:
    """Remove only one known disposable clone and its SQLite sidecars."""

    path.unlink(missing_ok=True)
    Path(f"{path}-wal").unlink(missing_ok=True)
    Path(f"{path}-shm").unlink(missing_ok=True)


def named_incorrect_option(presentation) -> Any:
    distractors = sorted(
        (option for option in presentation.question.options if not option.correct),
        key=lambda option: option.id,
    )
    named = [option for option in distractors if option.misconception_id]
    require(bool(named or distractors), "Selected question has no distractor.")
    return (named or distractors)[0]


def deterministic_answer(presentation, index: int) -> dict[str, Any]:
    """Mix credible, incorrect, uncertain, fast, and assisted observations."""

    pattern = index % 6
    if pattern == 0:
        selected = presentation.question.correct_option.id
        return {
            "selected_option_id": selected,
            "confidence": 0.92,
            "response_ms": 4200,
            "hint_count": 0,
        }
    if pattern == 1:
        selected = named_incorrect_option(presentation).id
        return {
            "selected_option_id": selected,
            "confidence": 0.93,
            "response_ms": 2600,
            "hint_count": 0,
        }
    if pattern == 2:
        return {
            "selected_option_id": None,
            "confidence": 0.20,
            "response_ms": 5400,
            "hint_count": 0,
        }
    if pattern == 3:
        selected = presentation.question.correct_option.id
        return {
            "selected_option_id": selected,
            "confidence": 0.30,
            "response_ms": 3600,
            "hint_count": 0,
        }
    if pattern == 4:
        selected = presentation.question.correct_option.id
        return {
            "selected_option_id": selected,
            "confidence": 0.96,
            "response_ms": 100,
            "hint_count": 0,
        }
    selected = named_incorrect_option(presentation).id
    return {
        "selected_option_id": selected,
        "confidence": 0.55,
        "response_ms": 7600,
        "hint_count": 1,
    }


def database_snapshot(database: Database, learner_id: str) -> dict[str, Any]:
    with database.read() as connection:
        counts = {
            table: connection.execute(
                f"SELECT COUNT(*) AS n FROM {table} WHERE learner_id = ?",
                (learner_id,),
            ).fetchone()["n"]
            for table in (
                "sessions",
                "decisions",
                "attempts",
                "skill_states",
                "objective_states",
                "objective_grid_states",
                "misconception_beliefs",
                "learner_skill_families",
                "learner_objective_families",
            )
        }
        posterior_sizes = [
            row["size_bytes"]
            for row in connection.execute(
                """SELECT length(posterior_blob) AS size_bytes
                   FROM objective_grid_states WHERE learner_id = ?
                   ORDER BY objective_id""",
                (learner_id,),
            )
        ]
        revision = connection.execute(
            "SELECT revision FROM learners WHERE id = ?", (learner_id,)
        ).fetchone()["revision"]
        events = connection.execute(
            "SELECT COUNT(*) AS n FROM events WHERE learner_id = ?",
            (learner_id,),
        ).fetchone()["n"]
        page_size = connection.execute("PRAGMA page_size").fetchone()[0]
        page_count = connection.execute("PRAGMA page_count").fetchone()[0]
    return {
        "learner_revision": revision,
        "learner_events": events,
        "counts": counts,
        "posterior_blob_bytes": {
            "count": len(posterior_sizes),
            "total": sum(posterior_sizes),
            "median": (
                round(statistics.median(posterior_sizes), 3)
                if posterior_sizes
                else 0
            ),
            "maximum": max(posterior_sizes, default=0),
        },
        "database_allocated_bytes": page_size * page_count,
    }


def build_base_database(path: Path, corpus: Path) -> None:
    database = Database(path)
    database.initialize()
    database.import_corpus(*read_and_parse(corpus, include_catalog=True))


def build_selection_template(
    base_path: Path,
    destination: Path,
    *,
    learner_id: str,
    history_responses: int,
    topic: str,
) -> dict[str, Any]:
    clone_database(base_path, destination)
    database = Database(destination)
    engine = AdaptiveEngine(database, LearnerModel(MODEL_VERSION))
    engine.create_learner(learner_id)
    session = engine.start_session(
        learner_id,
        topic,
        mode="learn",
        explore_related=False,
        seed=1701,
        now=START,
    )
    resets = 0
    build_started = time.perf_counter_ns()
    for index in range(history_responses):
        selected_at = START + timedelta(hours=6 * index)
        try:
            presentation = engine.next_question(session["id"], now=selected_at)
        except ExhaustedError:
            resets += 1
            require(resets <= 8, "History construction exhausted too many sessions.")
            session = engine.start_session(
                learner_id,
                topic,
                mode="learn",
                explore_related=False,
                seed=1701 + resets,
                now=selected_at,
            )
            presentation = engine.next_question(session["id"], now=selected_at)
        answer = deterministic_answer(presentation, index)
        engine.submit_answer(
            presentation.decision_id,
            answer["selected_option_id"],
            confidence=answer["confidence"],
            response_ms=answer["response_ms"],
            hint_count=answer["hint_count"],
            feedback_shown=False,
            idempotency_key=f"performance-history-{learner_id}-{index:03d}",
            now=selected_at + timedelta(minutes=1),
        )
    benchmark_time = START + timedelta(hours=6 * history_responses + 1)
    benchmark_session = session
    if history_responses:
        # A fresh session removes per-session recency as a confound while
        # retaining the learner's durable history-rich projection.
        benchmark_session = engine.start_session(
            learner_id,
            topic,
            mode="learn",
            explore_related=False,
            seed=9901,
            now=benchmark_time,
        )
    build_seconds = (time.perf_counter_ns() - build_started) / 1_000_000_000.0
    integrity = database.verify_integrity()
    require(integrity["ok"], "Selection template integrity failed: " + repr(integrity["errors"][:3]))
    return {
        "learner_id": learner_id,
        "session_id": benchmark_session["id"],
        "operation_time": benchmark_time,
        "history_responses_requested": history_responses,
        "history_session_resets": resets,
        "history_build_seconds": round(build_seconds, 6),
        "snapshot": database_snapshot(database, learner_id),
    }


def build_update_template(
    selection_template: Path,
    destination: Path,
    condition: Mapping[str, Any],
) -> dict[str, Any]:
    clone_database(selection_template, destination)
    database = Database(destination)
    engine = AdaptiveEngine(database, LearnerModel(MODEL_VERSION))
    presentation = engine.next_question(
        condition["session_id"], now=condition["operation_time"]
    )
    require(
        presentation.question.objective_id is not None,
        "Learner update benchmark selected a question without a learning objective.",
    )
    integrity = database.verify_integrity()
    require(integrity["ok"], "Update template integrity failed: " + repr(integrity["errors"][:3]))
    return {
        **condition,
        "decision_id": presentation.decision_id,
        "question_id": presentation.question.id,
        "objective_id": presentation.question.objective_id,
        "correct_option_id": presentation.question.correct_option.id,
        "incorrect_option_id": named_incorrect_option(presentation).id,
        "answer_time": condition["operation_time"] + timedelta(minutes=1),
    }


def benchmark_mutation(
    template: Path,
    scratch: Path,
    *,
    samples: int,
    warmups: int,
    label: str,
    operation: Callable[[AdaptiveEngine], Any],
) -> dict[str, Any]:
    observations: list[int] = []
    for index in range(warmups + samples):
        clone = scratch / f"{label}-{index:03d}.db"
        clone_database(template, clone)
        engine = AdaptiveEngine(Database(clone), LearnerModel(MODEL_VERSION))
        gc.collect()
        started = time.perf_counter_ns()
        operation(engine)
        elapsed = time.perf_counter_ns() - started
        if index >= warmups:
            observations.append(elapsed)
        remove_database(clone)
    return latency_summary(observations)


def benchmark_read_only(
    *,
    samples: int,
    warmups: int,
    operation: Callable[[], Any],
) -> dict[str, Any]:
    observations: list[int] = []
    for index in range(warmups + samples):
        gc.collect()
        started = time.perf_counter_ns()
        operation()
        elapsed = time.perf_counter_ns() - started
        if index >= warmups:
            observations.append(elapsed)
    return latency_summary(observations)


def profile_call(operation: Callable[[], Any], *, top: int) -> dict[str, Any]:
    profiler = cProfile.Profile()
    profiler.enable()
    operation()
    profiler.disable()
    stats = pstats.Stats(profiler)
    entries = []
    for (filename, line, function), values in sorted(
        stats.stats.items(), key=lambda item: item[1][3], reverse=True
    )[:top]:
        primitive_calls, total_calls, total_time, cumulative_time, _ = values
        try:
            rendered_file = str(Path(filename).resolve().relative_to(PROJECT_ROOT))
        except (OSError, ValueError):
            rendered_file = filename
        entries.append(
            {
                "file": rendered_file,
                "line": line,
                "function": function,
                "primitive_calls": primitive_calls,
                "total_calls": total_calls,
                "self_ms": round(total_time * 1000.0, 6),
                "cumulative_ms": round(cumulative_time * 1000.0, 6),
            }
        )
    return {
        "profile_total_ms": round(stats.total_tt * 1000.0, 6),
        "top_by_cumulative_time": entries,
    }


def profile_mutation(
    template: Path,
    scratch: Path,
    *,
    label: str,
    operation: Callable[[AdaptiveEngine], Any],
    top: int,
) -> dict[str, Any]:
    clone = scratch / f"profile-{label}.db"
    clone_database(template, clone)
    engine = AdaptiveEngine(Database(clone), LearnerModel(MODEL_VERSION))
    try:
        return profile_call(lambda: operation(engine), top=top)
    finally:
        remove_database(clone)


def ratio(numerator: Mapping[str, Any], denominator: Mapping[str, Any]) -> float:
    baseline = float(denominator["median_ms"])
    return round(float(numerator["median_ms"]) / baseline, 4) if baseline else 0.0


def run_lab(
    *,
    corpus: Path = DEFAULT_CORPUS,
    protected_database: Path = DEFAULT_PROTECTED_DATABASE,
    topic: str = DEFAULT_TOPIC,
    history_responses: int = DEFAULT_HISTORY_RESPONSES,
    samples: int = DEFAULT_SAMPLES,
    warmups: int = DEFAULT_WARMUPS,
    profile_top: int = DEFAULT_PROFILE_TOP,
) -> dict[str, Any]:
    require(corpus.exists(), f"Corpus does not exist: {corpus}")
    require(
        0 <= history_responses <= MAX_HISTORY_RESPONSES,
        f"history_responses must be in [0, {MAX_HISTORY_RESPONSES}].",
    )
    require(1 <= samples <= MAX_SAMPLES, f"samples must be in [1, {MAX_SAMPLES}].")
    require(0 <= warmups <= MAX_WARMUPS, f"warmups must be in [0, {MAX_WARMUPS}].")
    require(
        0 <= profile_top <= MAX_PROFILE_TOP,
        f"profile_top must be in [0, {MAX_PROFILE_TOP}].",
    )
    protected_before = database_family_fingerprint(protected_database)
    started = time.perf_counter_ns()

    with tempfile.TemporaryDirectory(prefix="tsq-learner-runtime-") as directory:
        scratch = Path(directory)
        base_path = scratch / "base.db"
        build_base_database(base_path, corpus)

        conditions: dict[str, dict[str, Any]] = {}
        selection_paths: dict[str, Path] = {}
        update_paths: dict[str, Path] = {}
        for name, history in (("cold", 0), ("rich", history_responses)):
            selection_path = scratch / f"{name}-selection.db"
            condition = build_selection_template(
                base_path,
                selection_path,
                learner_id=f"performance-{name}",
                history_responses=history,
                topic=topic,
            )
            update_path = scratch / f"{name}-update.db"
            update = build_update_template(
                selection_path, update_path, condition
            )
            conditions[name] = update
            selection_paths[name] = selection_path
            update_paths[name] = update_path

        require(
            conditions["rich"]["snapshot"]["counts"]["objective_grid_states"] > 0,
            "History-rich template did not exercise exact objective persistence.",
        )

        timings: dict[str, dict[str, Any]] = {}
        for name in ("cold", "rich"):
            condition = conditions[name]
            timings[f"selection_{name}"] = benchmark_mutation(
                selection_paths[name],
                scratch,
                samples=samples,
                warmups=warmups,
                label=f"selection-{name}",
                operation=lambda engine, condition=condition: engine.next_question(
                    condition["session_id"], now=condition["operation_time"]
                ),
            )
            for outcome, option_field in (
                ("correct", "correct_option_id"),
                ("incorrect", "incorrect_option_id"),
            ):
                timings[f"submission_{outcome}_{name}"] = benchmark_mutation(
                    update_paths[name],
                    scratch,
                    samples=samples,
                    warmups=warmups,
                    label=f"submission-{outcome}-{name}",
                    operation=(
                        lambda engine, condition=condition, option_field=option_field, outcome=outcome: engine.submit_answer(
                            condition["decision_id"],
                            condition[option_field],
                            confidence=0.90,
                            response_ms=1800,
                            hint_count=0,
                            feedback_shown=False,
                            idempotency_key=f"performance-{outcome}",
                            now=condition["answer_time"],
                        )
                    ),
                )

        rich_database = Database(selection_paths["rich"])
        rich_learner = conditions["rich"]["learner_id"]
        timings["objective_state_load_rich"] = benchmark_read_only(
            samples=samples,
            warmups=warmups,
            operation=lambda: rich_database.get_objective_states(rich_learner),
        )
        timings["projection_hash_v3_rich"] = benchmark_read_only(
            samples=samples,
            warmups=warmups,
            operation=lambda: rich_database.learner_projection_hash(
                rich_learner, hash_version=3
            ),
        )
        timings["integrity_check_rich"] = benchmark_read_only(
            samples=samples,
            warmups=warmups,
            operation=rich_database.verify_integrity,
        )

        profiles: dict[str, Any] = {}
        if profile_top:
            cold = conditions["cold"]
            rich = conditions["rich"]
            profiles["selection_cold"] = profile_mutation(
                selection_paths["cold"],
                scratch,
                label="selection-cold",
                operation=lambda engine: engine.next_question(
                    cold["session_id"], now=cold["operation_time"]
                ),
                top=profile_top,
            )
            profiles["selection_rich"] = profile_mutation(
                selection_paths["rich"],
                scratch,
                label="selection-rich",
                operation=lambda engine: engine.next_question(
                    rich["session_id"], now=rich["operation_time"]
                ),
                top=profile_top,
            )
            profiles["submission_correct_cold"] = profile_mutation(
                update_paths["cold"],
                scratch,
                label="submission-cold",
                operation=lambda engine: engine.submit_answer(
                    cold["decision_id"],
                    cold["correct_option_id"],
                    confidence=0.90,
                    response_ms=1800,
                    hint_count=0,
                    feedback_shown=False,
                    idempotency_key="performance-profile-correct-cold",
                    now=cold["answer_time"],
                ),
                top=profile_top,
            )
            profiles["submission_correct_rich"] = profile_mutation(
                update_paths["rich"],
                scratch,
                label="submission-rich",
                operation=lambda engine: engine.submit_answer(
                    rich["decision_id"],
                    rich["correct_option_id"],
                    confidence=0.90,
                    response_ms=1800,
                    hint_count=0,
                    feedback_shown=False,
                    idempotency_key="performance-profile-correct",
                    now=rich["answer_time"],
                ),
                top=profile_top,
            )
            profiles["objective_state_load_rich"] = profile_call(
                lambda: rich_database.get_objective_states(rich_learner),
                top=profile_top,
            )
            profiles["projection_hash_v3_rich"] = profile_call(
                lambda: rich_database.learner_projection_hash(
                    rich_learner, hash_version=3
                ),
                top=profile_top,
            )

        report = {
            "lab_version": LAB_VERSION,
            "environment": {
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "sqlite": sqlite3.sqlite_version,
            },
            "contracts": {
                "schema_version": SCHEMA_VERSION,
                "learner_model_version": MODEL_VERSION,
                "policy_version": POLICY_VERSION,
                "posterior_algorithm": OBJECTIVE_POSTERIOR_ALGORITHM,
                "posterior_grid_points": GRID_SIZE,
            },
            "config": {
                "corpus": str(corpus.resolve()),
                "corpus_sha256": corpus_digest(corpus),
                "topic": topic,
                "history_responses": history_responses,
                "samples": samples,
                "warmups": warmups,
                "profile_top": profile_top,
                "feedback_shown": False,
            },
            "conditions": {
                name: {
                    key: value.isoformat()
                    if isinstance(value, datetime)
                    else value
                    for key, value in condition.items()
                }
                for name, condition in conditions.items()
            },
            "timings": timings,
            "comparisons": {
                "selection_rich_over_cold_median": ratio(
                    timings["selection_rich"], timings["selection_cold"]
                ),
                "correct_submission_rich_over_cold_median": ratio(
                    timings["submission_correct_rich"],
                    timings["submission_correct_cold"],
                ),
                "incorrect_submission_rich_over_cold_median": ratio(
                    timings["submission_incorrect_rich"],
                    timings["submission_incorrect_cold"],
                ),
            },
            "profiles": profiles,
            "methodology": {
                "clock": "time.perf_counter_ns",
                "percentile": "nearest-rank p95",
                "mutation_isolation": (
                    "Each timed mutation starts from a fresh SQLite backup; "
                    "backup creation and cleanup are outside the timed interval."
                ),
                "read_measurements": (
                    "Repeated against one immutable warm-cache template."
                ),
                "history_pattern": (
                    "Fixed six-response cycle: deliberate correct, confident "
                    "named misconception, uncertain abstention, low-confidence "
                    "correct, implausibly fast correct, hinted incorrect."
                ),
                "interpretation": (
                    "Absolute latency is machine-specific; compare medians, p95s, "
                    "condition ratios, and cumulative profile shares."
                ),
            },
            "temporary_databases_destroyed": True,
        }

    protected_after = database_family_fingerprint(protected_database)
    report["protected_database"] = {
        "before": protected_before,
        "after": protected_after,
        "unchanged": protected_before == protected_after,
        "opened_by_tsq": False,
    }
    require(
        report["protected_database"]["unchanged"],
        "Protected database changed during the performance laboratory.",
    )
    report["wall_seconds"] = round(
        (time.perf_counter_ns() - started) / 1_000_000_000.0, 6
    )
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    result.add_argument(
        "--protected-db", type=Path, default=DEFAULT_PROTECTED_DATABASE
    )
    result.add_argument("--topic", default=DEFAULT_TOPIC)
    result.add_argument(
        "--history-responses", type=int, default=DEFAULT_HISTORY_RESPONSES
    )
    result.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    result.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    result.add_argument("--profile-top", type=int, default=DEFAULT_PROFILE_TOP)
    result.add_argument(
        "--output",
        type=Path,
        help="Optional JSON destination. The report is otherwise written to stdout.",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    report = run_lab(
        corpus=arguments.corpus,
        protected_database=arguments.protected_db,
        topic=arguments.topic,
        history_responses=arguments.history_responses,
        samples=arguments.samples,
        warmups=arguments.warmups,
        profile_top=arguments.profile_top,
    )
    encoded = json.dumps(
        report, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    if arguments.output is None:
        sys.stdout.write(encoded)
    else:
        destination = arguments.output.resolve()
        require(
            destination != arguments.protected_db.resolve(),
            "Output must not overwrite the protected database.",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded, encoding="utf-8")
        sys.stdout.write(
            json.dumps(
                {
                    "lab_version": LAB_VERSION,
                    "output": str(destination),
                    "protected_database_unchanged": report[
                        "protected_database"
                    ]["unchanged"],
                    "wall_seconds": report["wall_seconds"],
                },
                sort_keys=True,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
