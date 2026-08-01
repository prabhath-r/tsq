# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from tsq.authoring import CoveragePlanner
from tsq.cli import _print_compact_study_completion, build_parser, main
from tsq.corpus import load_bundle, read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.errors import ConflictError
from tsq.store import SCHEMA_VERSION, Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"


def durable_database_fingerprint(path: Path) -> dict[str, tuple[int, str] | None]:
    """Hash durable SQLite content while normalizing an empty WAL to absence."""

    result: dict[str, tuple[int, str] | None] = {}
    for label, candidate in (
        ("main", path),
        ("wal", Path(f"{path}-wal")),
        ("journal", Path(f"{path}-journal")),
    ):
        if not candidate.exists() or (
            label in {"wal", "journal"} and candidate.stat().st_size == 0
        ):
            result[label] = None
            continue
        material = candidate.read_bytes()
        result[label] = (len(material), hashlib.sha256(material).hexdigest())
    return result


def logical_database_snapshot(database: Database) -> dict:
    with database.read() as connection:
        tables = tuple(
            row["name"]
            for row in connection.execute(
                """SELECT name FROM sqlite_master
                   WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                   ORDER BY name"""
            ).fetchall()
        )
        return {
            "schema": tuple(
                (row["type"], row["name"], row["sql"])
                for row in connection.execute(
                    """SELECT type, name, sql FROM sqlite_master
                       WHERE name NOT LIKE 'sqlite_%'
                       ORDER BY type, name"""
                ).fetchall()
            ),
            "meta": tuple(
                tuple(row)
                for row in connection.execute(
                    "SELECT key, value FROM meta ORDER BY key"
                ).fetchall()
            ),
            "table_counts": tuple(
                (
                    table,
                    connection.execute(
                        f'SELECT COUNT(*) AS n FROM "{table}"'
                    ).fetchone()["n"],
                )
                for table in tables
            ),
            "event_tail": tuple(
                tuple(row)
                for row in connection.execute(
                    """SELECT event_id, stream_id, stream_version, event_type,
                              payload_hash
                       FROM events
                       ORDER BY recorded_at DESC, event_id DESC LIMIT 20"""
                ).fetchall()
            ),
            "decision_invalidations": connection.execute(
                """SELECT COUNT(*) AS n FROM events
                   WHERE event_type = 'DecisionInvalidated'"""
            ).fetchone()["n"],
        }


def backup_database(source: Database, target: Path) -> None:
    """Create a transactionally consistent fixture, including visible WAL data."""

    with source.read() as source_connection, closing(
        sqlite3.connect(target)
    ) as target_connection:
        source_connection.backup(target_connection)
        target_connection.commit()


class CliJourneyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.tempdir.name) / "cli.db")
        self.database.initialize()
        self.database.import_corpus(
            *read_and_parse(CORPUS, include_catalog=True)
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def run_json(self, *arguments: str):
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                ["--db", str(self.database.path), *arguments, "--json"]
            )
        self.assertEqual(exit_code, 0)
        return json.loads(output.getvalue())

    def test_strict_audit_keeps_quarantined_only_mapping_gaps_visible(
        self,
    ) -> None:
        bundle = load_bundle(CORPUS)
        question = next(
            item
            for item in bundle["questions"]
            if item["status"] == "quarantined"
            and item.get("provenance", {}).get("generated") is True
        )
        primary_concept_id = next(
            mapping["concept_id"]
            for mapping in question["concepts"]
            if mapping["role"] == "primary"
        )
        new_concept_id = "c_quarantined_support_only"
        bundle["concepts"].append(
            {
                "id": new_concept_id,
                "name": "Quarantined support only",
                "description": (
                    "A test-only concept mapped by quarantined evidence but "
                    "never by an active primary item."
                ),
                "prior_mastery": 0.2,
            }
        )
        owner = next(
            topic
            for topic in bundle["topics"]
            if primary_concept_id in topic["concept_ids"]
        )
        owner["concept_ids"].append(new_concept_id)
        for mapping in question["concepts"]:
            mapping["weight"] *= 0.9
        question["concepts"].append(
            {
                "concept_id": new_concept_id,
                "weight": 0.1,
                "role": "supporting",
            }
        )
        corpus_path = Path(self.tempdir.name) / "warning-corpus.json"
        corpus_path.write_text(json.dumps(bundle), encoding="utf-8")
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main(
                ["audit", str(corpus_path), "--strict", "--json"]
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertIn(
            "missing_primary_mapping_coverage",
            {issue["code"] for issue in payload["warnings"]},
        )

    def test_interactive_commands_collect_confidence_by_default(self) -> None:
        parser = build_parser()

        self.assertTrue(parser.parse_args(["start"]).ask_confidence)
        self.assertFalse(
            parser.parse_args(["start", "--no-confidence"]).ask_confidence
        )
        self.assertTrue(
            parser.parse_args(
                ["study", "--learner", "me", "--topic", "Transformers"]
            ).ask_confidence
        )
        self.assertFalse(
            parser.parse_args(
                [
                    "study",
                    "--learner",
                    "me",
                    "--topic",
                    "Transformers",
                    "--no-confidence",
                ]
            ).ask_confidence
        )
        self.assertFalse(parser.parse_args(["start"]).details)
        self.assertTrue(parser.parse_args(["start", "--details"]).details)
        self.assertFalse(
            parser.parse_args(
                ["study", "--learner", "me", "--topic", "Transformers"]
            ).details
        )
        self.assertTrue(
            parser.parse_args(
                [
                    "study",
                    "--learner",
                    "me",
                    "--topic",
                    "Transformers",
                    "--details",
                ]
            ).details
        )

    def test_interactive_abstention_skips_confidence_and_is_neutral(self) -> None:
        prompts: list[str] = []

        def answer(prompt: str) -> str:
            prompts.append(prompt)
            if prompt == "answer> ":
                return "?"
            self.fail(f"Unexpected prompt after abstention: {prompt}")

        output = io.StringIO()
        error = io.StringIO()
        with patch("builtins.input", side_effect=answer), redirect_stdout(
            output
        ), redirect_stderr(error):
            exit_code = main(
                [
                    "--db",
                    str(self.database.path),
                    "study",
                    "--learner",
                    "cli-abstention",
                    "--topic",
                    "LLM Agents",
                    "--limit",
                    "1",
                    "--seed",
                    "17",
                ]
            )

        self.assertEqual(exit_code, 0, error.getvalue())
        self.assertEqual(prompts, ["answer> "])
        rendered = output.getvalue()
        self.assertIn("— Skipped — you chose 'I do not know'", rendered)
        self.assertNotIn("✗ Not correct", rendered)
        self.assertIn("[PRACTICE]", rendered)
        self.assertNotIn("[LEARN]", rendered)
        self.assertIn("Session evidence:", rendered)
        self.assertIn("0 selected answers · 1 skipped", rendered)
        self.assertIn("1 completed · 0 correct · 0 incorrect · 1 skipped", rendered)
        self.assertIn(
            "Evidence recorded for: Apply tool authorization boundaries",
            rendered,
        )
        self.assertIn(
            "Next focus: Apply tool authorization boundaries",
            rendered,
        )
        self.assertNotIn("Objective projection:", rendered)
        self.assertNotIn("(lo_", rendered)
        self.assertNotIn("marked unsure", rendered)
        self.assertNotIn("`tsq session report", rendered)
        self.assertIn("in the same database", rendered)
        self.assertIn("provisional session signals", rendered)
        self.assertNotIn("fine-grained objective evidence:", rendered)
        self.assertNotIn(
            "fine-grained selected-response evidence:",
            rendered,
        )
        self.assertNotIn("Assessed selected-response objectives:", rendered)
        with self.database.read() as connection:
            attempt = connection.execute(
                """SELECT selected_option_id, confidence
                   FROM attempts
                   WHERE learner_id = 'cli-abstention'"""
            ).fetchone()
        self.assertIsNotNone(attempt)
        self.assertIsNone(attempt["selected_option_id"])
        self.assertIsNone(attempt["confidence"])

    def test_interactive_selected_answer_still_requests_confidence(self) -> None:
        prompts: list[str] = []
        responses = iter(("1", ""))

        def answer(prompt: str) -> str:
            prompts.append(prompt)
            return next(responses)

        output = io.StringIO()
        error = io.StringIO()
        with patch("builtins.input", side_effect=answer), redirect_stdout(
            output
        ), redirect_stderr(error):
            exit_code = main(
                [
                    "--db",
                    str(self.database.path),
                    "study",
                    "--learner",
                    "cli-selected",
                    "--topic",
                    "LLM Agents",
                    "--limit",
                    "1",
                    "--seed",
                    "18",
                ]
            )

        self.assertEqual(exit_code, 0, error.getvalue())
        self.assertEqual(
            prompts,
            [
                "answer> ",
                "confidence 0-100 (blank to skip)> ",
            ],
        )
        with self.database.read() as connection:
            attempt = connection.execute(
                """SELECT selected_option_id, confidence
                   FROM attempts WHERE learner_id = 'cli-selected'"""
            ).fetchone()
        self.assertIsNotNone(attempt["selected_option_id"])
        self.assertIsNone(attempt["confidence"])

    def test_interactive_details_preserve_internal_explanations(self) -> None:
        output = io.StringIO()
        error = io.StringIO()
        with patch("builtins.input", return_value="?"), redirect_stdout(
            output
        ), redirect_stderr(error):
            exit_code = main(
                [
                    "--db",
                    str(self.database.path),
                    "study",
                    "--learner",
                    "cli-details",
                    "--topic",
                    "LLM Agents",
                    "--limit",
                    "1",
                    "--seed",
                    "19",
                    "--explain-policy",
                    "--details",
                ]
            )

        self.assertEqual(exit_code, 0, error.getvalue())
        rendered = output.getvalue()
        self.assertIn("policy: phase=", rendered)
        self.assertIn("pedagogical_role=main", rendered)
        self.assertIn("objective internals: id=", rendered)
        self.assertIn("Objective projection:", rendered)
        self.assertIn("Next probe objective:", rendered)
        self.assertIn("inference boundary:", rendered)
        self.assertIn("1 completed · 0 correct · 0 incorrect · 1 skipped", rendered)
        self.assertIn("fine-grained selected-response evidence:", rendered)
        self.assertIn("0 selected answers · 1 skipped", rendered)
        self.assertIn("Learner: cli-details", rendered)

    def test_compact_summary_separates_mixed_selected_answers_and_skips(
        self,
    ) -> None:
        report = {
            "session_id": "ses_mixed",
            "topic": {"id": "t_example", "name": "Example"},
            "root_concept_id": "c_example",
            "status": "completed",
            "questions_answered": 10,
            "correct": 1,
            "selected_answers": 3,
            "selected_incorrect": 2,
            "selected_accuracy": 1 / 3,
            "abstained": 7,
            "remediation_questions": 2,
            "objective_performance": [
                {
                    "name": "Example objective",
                    "session": {
                        "correct": 1,
                        "selected_answers": 3,
                        "abstained": 7,
                    },
                }
            ],
        }
        output = io.StringIO()
        with redirect_stdout(output):
            _print_compact_study_completion(report)
        rendered = output.getvalue()
        self.assertIn(
            "10 completed · 1 correct · 2 incorrect · 7 skipped",
            rendered,
        )
        self.assertIn(
            "Among selected answers: 1/3 correct (33.3%)",
            rendered,
        )
        self.assertIn(
            "Example objective: 1/3 selected answers correct · 7 skipped",
            rendered,
        )
        self.assertIn("For full evidence on future sessions", rendered)

    def test_session_list_recovers_history_with_filters_and_stable_order(self) -> None:
        engine = AdaptiveEngine(self.database)
        engine.create_learner("alpha", "Alpha Learner")
        engine.create_learner("beta", "Beta Learner")
        started_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
        completed = engine.start_session(
            "alpha", "Transformers", seed=11, now=started_at
        )
        presentation = engine.next_question(
            completed["id"], now=started_at
        )
        engine.submit_answer(
            presentation.decision_id,
            presentation.question.correct_option.id,
            confidence=0.9,
            response_ms=1_200,
            now=started_at + timedelta(milliseconds=1_200),
        )
        engine.end_session(
            completed["id"],
            status="completed",
            now=started_at + timedelta(seconds=2),
        )
        active = engine.start_session("alpha", "LLM Agents", seed=13)
        other = engine.start_session("beta", "Transformers", seed=17)

        tied_timestamp = "2100-01-01T00:00:00+00:00"
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE learner_id = 'alpha'",
                (tied_timestamp,),
            )

        history = self.run_json("session", "list", "--learner", "alpha")
        self.assertEqual(
            [row["id"] for row in history],
            sorted((completed["id"], active["id"]), reverse=True),
        )
        by_id = {row["id"]: row for row in history}
        self.assertEqual(by_id[completed["id"]]["status"], "completed")
        self.assertEqual(by_id[completed["id"]]["questions_answered"], 1)
        self.assertEqual(by_id[completed["id"]]["correct"], 1)
        self.assertEqual(by_id[completed["id"]]["accuracy"], 1.0)
        self.assertEqual(by_id[completed["id"]]["selected_answers"], 1)
        self.assertEqual(by_id[completed["id"]]["selected_incorrect"], 0)
        self.assertEqual(by_id[completed["id"]]["selected_accuracy"], 1.0)
        self.assertEqual(by_id[active["id"]]["target_name"], "LLM Agents")
        self.assertNotIn(other["id"], by_id)

        filtered = self.run_json(
            "session",
            "list",
            "--learner",
            "alpha",
            "--status",
            "completed",
            "--limit",
            "1",
        )
        self.assertEqual([row["id"] for row in filtered], [completed["id"]])

    def test_session_list_rejects_unbounded_or_blank_filters_cleanly(self) -> None:
        for arguments, message in (
            (("--limit", "0"), "from 1 to 200"),
            (("--limit", "201"), "from 1 to 200"),
            (("--learner", ""), "must not be blank"),
        ):
            with self.subTest(arguments=arguments):
                error = io.StringIO()
                with redirect_stderr(error):
                    exit_code = main(
                        [
                            "--db",
                            str(self.database.path),
                            "session",
                            "list",
                            *arguments,
                        ]
                    )
                self.assertEqual(exit_code, 2)
                self.assertIn(message, error.getvalue())
                self.assertNotIn("Traceback", error.getvalue())

    def test_answer_records_feedback_only_after_cli_output(self) -> None:
        engine = AdaptiveEngine(self.database)
        engine.create_learner("feedback-cli")
        session = engine.start_session("feedback-cli", "Transformers", seed=19)
        presentation = engine.next_question(session["id"])
        arguments = (
            "answer",
            presentation.decision_id,
            presentation.question.correct_option.id,
            "--confidence",
            "0.9",
            "--response-ms",
            "0",
            "--idempotency-key",
            "cli-feedback-answer",
        )

        first = self.run_json(*arguments)
        second = self.run_json(*arguments)

        self.assertTrue(first["correct"])
        self.assertEqual(first["outcome"], "correct")
        self.assertTrue(second["idempotent_replay"])
        self.assertEqual(second["interaction_id"], first["interaction_id"])
        with self.database.read() as connection:
            attempt = connection.execute(
                "SELECT feedback_shown FROM attempts WHERE decision_id = ?",
                (presentation.decision_id,),
            ).fetchone()
            actions = connection.execute(
                """SELECT stage, action_type, payload_json
                   FROM learning_actions WHERE decision_id = ?""",
                (presentation.decision_id,),
            ).fetchall()
        self.assertEqual(attempt["feedback_shown"], 0)
        self.assertEqual(len(actions), 1)
        self.assertEqual(
            (actions[0]["stage"], actions[0]["action_type"]),
            ("post_feedback", "feedback_shown"),
        )
        self.assertEqual(
            len(json.loads(actions[0]["payload_json"])["feedback_digest"]),
            64,
        )

        next_presentation = engine.next_question(session["id"])
        wrong_option = next(
            option
            for option in next_presentation.question.options
            if not option.correct
        )
        incorrect = self.run_json(
            "answer",
            next_presentation.decision_id,
            wrong_option.id,
            "--confidence",
            "0.9",
            "--response-ms",
            "0",
            "--idempotency-key",
            "cli-feedback-incorrect",
        )
        self.assertFalse(incorrect["correct"])
        self.assertEqual(incorrect["outcome"], "incorrect")

    def test_answer_command_preserves_abstention_as_a_distinct_outcome(self) -> None:
        engine = AdaptiveEngine(self.database)
        engine.create_learner("answer-abstention")
        session = engine.start_session(
            "answer-abstention", "LLM Agents", seed=23
        )
        presentation = engine.next_question(session["id"])

        payload = self.run_json(
            "answer",
            presentation.decision_id,
            "?",
            "--confidence",
            "0.99",
            "--response-ms",
            "0",
            "--idempotency-key",
            "answer-abstention-json",
        )

        self.assertEqual(payload["outcome"], "abstained")
        self.assertFalse(payload["correct"])
        self.assertIsNone(payload["selected_option_id"])
        with self.database.read() as connection:
            attempt = connection.execute(
                """SELECT selected_option_id, confidence
                   FROM attempts WHERE decision_id = ?""",
                (presentation.decision_id,),
            ).fetchone()
        self.assertIsNotNone(attempt)
        self.assertIsNone(attempt["selected_option_id"])
        self.assertIsNone(attempt["confidence"])

        second = engine.next_question(session["id"])
        output = io.StringIO()
        error = io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            exit_code = main(
                [
                    "--db",
                    str(self.database.path),
                    "answer",
                    second.decision_id,
                    "unknown",
                    "--confidence",
                    "0.75",
                    "--response-ms",
                    "0",
                    "--idempotency-key",
                    "answer-abstention-text",
                ]
            )
        self.assertEqual(exit_code, 0, error.getvalue())
        rendered = output.getvalue()
        self.assertIn("Skipped — you chose 'I do not know'.", rendered)
        self.assertNotIn("Not correct.", rendered)

    def test_answer_command_replays_legacy_abstention_confidence(self) -> None:
        engine = AdaptiveEngine(self.database)
        engine.create_learner("legacy-abstention")
        session = engine.start_session(
            "legacy-abstention", "LLM Agents", seed=29
        )
        presentation = engine.next_question(session["id"])
        original = engine.submit_answer(
            presentation.decision_id,
            None,
            confidence=0.99,
            response_ms=0,
            feedback_shown=False,
            idempotency_key="legacy-abstention-retry",
        )

        replay = self.run_json(
            "answer",
            presentation.decision_id,
            "?",
            "--confidence",
            "0.99",
            "--response-ms",
            "0",
            "--idempotency-key",
            "legacy-abstention-retry",
        )

        self.assertEqual(replay["outcome"], "abstained")
        self.assertTrue(replay["idempotent_replay"])
        self.assertEqual(replay["interaction_id"], original.interaction_id)
        with self.database.read() as connection:
            attempt = connection.execute(
                """SELECT confidence FROM attempts
                   WHERE decision_id = ?""",
                (presentation.decision_id,),
            ).fetchone()
        self.assertEqual(attempt["confidence"], 0.99)

    def test_topics_distinguish_graph_concepts_from_learning_objectives(self) -> None:
        catalog = self.run_json("topics")
        transformer = next(
            topic for topic in catalog["topics"] if topic["id"] == "t_transformers"
        )
        stored_topic = self.database.resolve_topic("t_transformers")
        owned_concepts = {row["id"] for row in stored_topic["concepts"]}
        release_objectives = self.database.get_learning_objectives(
            stored_topic["release_id"]
        )
        owned_objectives = [
            objective
            for objective in release_objectives
            if objective.primary_concept_id in owned_concepts
        ]
        self.assertEqual(transformer["direct_concepts"], len(owned_concepts))
        self.assertEqual(
            transformer["direct_learning_objectives"], len(owned_objectives)
        )
        self.assertGreaterEqual(
            transformer["scope_concepts"], transformer["direct_concepts"]
        )
        self.assertGreaterEqual(
            transformer["scope_learning_objectives"],
            transformer["direct_learning_objectives"],
        )
        self.assertGreater(
            transformer["direct_learning_objectives"],
            transformer["direct_concepts"],
        )

        graph = self.run_json("graph", "Transformers")
        graph_scope = self.database.topic_scope(
            "t_transformers", stored_topic["release_id"]
        )
        graph_objectives = [
            objective
            for objective in release_objectives
            if objective.primary_concept_id in graph_scope
        ]
        self.assertEqual(len(graph["concepts"]), len(graph_scope))
        self.assertEqual(
            len(graph["learning_objectives"]),
            len(graph_objectives),
        )
        self.assertTrue(
            all(
                objective["primary_concept_id"]
                in {concept["id"] for concept in graph["concepts"]}
                for objective in graph["learning_objectives"]
            )
        )

    def test_inspection_commands_preserve_all_logical_and_durable_state(
        self,
    ) -> None:
        engine = AdaptiveEngine(self.database)
        engine.create_learner("readonly-cli", "Read Only")
        session = engine.start_session(
            "readonly-cli", "Transformers", seed=101
        )
        presentation = engine.next_question(session["id"])
        gaps = CoveragePlanner(self.database).gaps(limit=1)
        self.assertTrue(gaps)
        job_id = CoveragePlanner(self.database).enqueue(gaps)[0]

        baseline_logical = logical_database_snapshot(self.database)
        commands = (
            ("topics",),
            ("topics", "--concepts"),
            ("graph", "Transformers"),
            ("session", "list", "--learner", "readonly-cli"),
            ("session", "report", session["id"]),
            ("action", "list", presentation.decision_id),
            ("profile", "--learner", "readonly-cli"),
            ("trace", session["id"]),
            ("coverage", "--limit", "1"),
            ("jobs", "list"),
            ("jobs", "show", job_id),
            ("reviews", "show", job_id),
            ("verify",),
        )
        for command in commands:
            with self.subTest(command=command):
                before = durable_database_fingerprint(self.database.path)
                output = io.StringIO()
                error = io.StringIO()
                with redirect_stdout(output), redirect_stderr(error):
                    exit_code = main(
                        [
                            "--db",
                            str(self.database.path),
                            *command,
                            "--json",
                        ]
                    )
                self.assertEqual(exit_code, 0, error.getvalue())
                json.loads(output.getvalue())
                self.assertEqual(
                    durable_database_fingerprint(self.database.path),
                    before,
                )

        self.assertEqual(
            logical_database_snapshot(self.database),
            baseline_logical,
        )

    def test_human_session_report_labels_position_shadow_non_certifying(
        self,
    ) -> None:
        engine = AdaptiveEngine(self.database)
        engine.create_learner("position-shadow-cli")
        started_at = datetime(2100, 1, 2, tzinfo=timezone.utc)
        session = engine.start_session(
            "position-shadow-cli",
            "Transformers",
            seed=71,
            now=started_at,
        )
        presentation = engine.next_question(session["id"], now=started_at)
        engine.submit_answer(
            presentation.decision_id,
            presentation.ordered_options[0].id,
            confidence=0.75,
            response_ms=1_200,
            now=started_at + timedelta(seconds=2),
        )

        output = io.StringIO()
        error = io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            exit_code = main(
                [
                    "--db",
                    str(self.database.path),
                    "session",
                    "report",
                    session["id"],
                ]
            )

        self.assertEqual(exit_code, 0, error.getvalue())
        rendered = output.getvalue()
        self.assertIn("response-position shadow: inconclusive", rendered)
        self.assertIn("shadow-only behavioral hypothesis", rendered)
        self.assertIn("does not certify skill", rendered)
        self.assertIn("provisional selected-response model", rendered)
        self.assertIn("not empirically validated", rendered)
        self.assertIn(
            "release-wide calibrated eligible items 0/", rendered
        )
        self.assertIn(
            "numerical guard covers approximation only", rendered
        )
        self.assertIn(
            "authored priors; no empirical calibration has been validated",
            rendered,
        )
        self.assertNotIn("until sufficient response data exists", rendered)

        profile_output = io.StringIO()
        profile_error = io.StringIO()
        with redirect_stdout(profile_output), redirect_stderr(profile_error):
            profile_exit = main(
                [
                    "--db",
                    str(self.database.path),
                    "profile",
                    "--learner",
                    "position-shadow-cli",
                    "--topic",
                    "Transformers",
                ]
            )
        self.assertEqual(profile_exit, 0, profile_error.getvalue())
        profile_rendered = profile_output.getvalue()
        self.assertIn("provisional selected-response model", profile_rendered)
        self.assertIn("not empirically validated", profile_rendered)
        self.assertIn(
            "release-wide calibrated eligible items 0/",
            profile_rendered,
        )
        self.assertIn(
            "numerical guard covers approximation only",
            profile_rendered,
        )

    def test_read_only_inspection_preserves_live_wal_and_events(self) -> None:
        writer = self.database.connect()
        try:
            writer.execute("PRAGMA wal_autocheckpoint = 0")
            writer.execute(
                "INSERT INTO meta(key, value) VALUES('readonly_probe', 'visible')"
            )
            writer.commit()
            wal_path = Path(f"{self.database.path}-wal")
            self.assertTrue(wal_path.exists())
            self.assertGreater(wal_path.stat().st_size, 0)
            before = durable_database_fingerprint(self.database.path)
            event_count = writer.execute(
                "SELECT COUNT(*) AS n FROM events"
            ).fetchone()["n"]
            invalidations = writer.execute(
                """SELECT COUNT(*) AS n FROM events
                   WHERE event_type = 'DecisionInvalidated'"""
            ).fetchone()["n"]

            read_only = Database(self.database.path, read_only=True)
            read_only.validate_current_schema()
            with read_only.read() as connection:
                probe = connection.execute(
                    "SELECT value FROM meta WHERE key = 'readonly_probe'"
                ).fetchone()
            self.assertIsNotNone(probe)
            self.assertEqual(probe["value"], "visible")
            self.assertEqual(
                durable_database_fingerprint(self.database.path), before
            )

            output = io.StringIO()
            error = io.StringIO()
            with redirect_stdout(output), redirect_stderr(error):
                exit_code = main(
                    [
                        "--db",
                        str(self.database.path),
                        "topics",
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 0, error.getvalue())
            self.assertTrue(json.loads(output.getvalue())["topics"])
            self.assertEqual(
                durable_database_fingerprint(self.database.path), before
            )
            self.assertEqual(
                writer.execute(
                    "SELECT COUNT(*) AS n FROM events"
                ).fetchone()["n"],
                event_count,
            )
            self.assertEqual(
                writer.execute(
                    """SELECT COUNT(*) AS n FROM events
                       WHERE event_type = 'DecisionInvalidated'"""
                ).fetchone()["n"],
                invalidations,
            )
        finally:
            writer.close()

    def test_inspection_fails_closed_without_creating_or_migrating(self) -> None:
        readonly = Database(self.database.path, read_only=True)
        with self.assertRaisesRegex(ConflictError, "cannot be initialized"):
            readonly.initialize()
        with self.assertRaisesRegex(ConflictError, "write transaction"):
            with readonly.transaction():
                self.fail("read-only transaction unexpectedly opened")
        with readonly.read() as connection:
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute(
                    "INSERT INTO meta(key, value) VALUES('forbidden', 'write')"
                )

        missing = Path(self.tempdir.name) / "absent" / "missing.db"
        for command in (("verify",), ("topics",)):
            with self.subTest(command=command):
                error = io.StringIO()
                with redirect_stderr(error):
                    exit_code = main(
                        ["--db", str(missing), *command, "--json"]
                    )
                self.assertEqual(exit_code, 2)
                self.assertIn("does not exist", error.getvalue())
                self.assertFalse(missing.exists())
                self.assertFalse(missing.parent.exists())

        no_db_command_path = (
            Path(self.tempdir.name) / "contracts" / "unused.db"
        )
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                [
                    "--db",
                    str(no_db_command_path),
                    "action",
                    "kinds",
                    "--json",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertTrue(json.loads(output.getvalue()))
        self.assertFalse(no_db_command_path.parent.exists())

        uninitialized = Path(self.tempdir.name) / "uninitialized.db"
        uninitialized.touch()
        before = durable_database_fingerprint(uninitialized)
        error = io.StringIO()
        with redirect_stderr(error):
            exit_code = main(
                ["--db", str(uninitialized), "verify", "--json"]
            )
        self.assertEqual(exit_code, 2)
        self.assertIn("no TSQ schema metadata", error.getvalue())
        self.assertEqual(durable_database_fingerprint(uninitialized), before)

        incompatible = Path(self.tempdir.name) / "incompatible.db"
        incompatible_database = Database(incompatible)
        incompatible_database.initialize()
        cases = (
            ("11", "explicit writable migration"),
            ("999", "supports at most"),
        )
        for version, message in cases:
            with incompatible_database.transaction() as connection:
                connection.execute(
                    """UPDATE meta SET value = ?
                       WHERE key = 'schema_version'""",
                    (version,),
                )
            before = durable_database_fingerprint(incompatible)
            error = io.StringIO()
            with redirect_stderr(error):
                exit_code = main(
                    ["--db", str(incompatible), "verify", "--json"]
                )
            self.assertEqual(exit_code, 2)
            self.assertIn(message, error.getvalue())
            self.assertEqual(
                durable_database_fingerprint(incompatible), before
            )

        with incompatible_database.transaction() as connection:
            connection.execute(
                """UPDATE meta SET value = ?
                   WHERE key = 'schema_version'""",
                (str(SCHEMA_VERSION),),
            )
            connection.execute("DROP TABLE item_reviews")
        before = durable_database_fingerprint(incompatible)
        error = io.StringIO()
        with redirect_stderr(error):
            exit_code = main(
                ["--db", str(incompatible), "verify", "--json"]
            )
        self.assertEqual(exit_code, 2)
        self.assertIn("missing tables: item_reviews", error.getvalue())
        self.assertEqual(durable_database_fingerprint(incompatible), before)

    def test_inspection_rejects_missing_or_malformed_schema_without_mutation(
        self,
    ) -> None:
        missing_trigger = Path(self.tempdir.name) / "missing-trigger.db"
        backup_database(self.database, missing_trigger)
        with closing(sqlite3.connect(missing_trigger)) as connection:
            connection.execute("DROP TRIGGER events_no_update")
            connection.commit()
        before = durable_database_fingerprint(missing_trigger)
        error = io.StringIO()
        with redirect_stderr(error):
            exit_code = main(
                ["--db", str(missing_trigger), "verify", "--json"]
            )
        self.assertEqual(exit_code, 2)
        self.assertIn("missing triggers: events_no_update", error.getvalue())
        self.assertNotIn("OperationalError", error.getvalue())
        self.assertNotIn("Traceback", error.getvalue())
        self.assertEqual(
            durable_database_fingerprint(missing_trigger),
            before,
        )

        changed_trigger = Path(self.tempdir.name) / "changed-trigger.db"
        backup_database(self.database, changed_trigger)
        with closing(sqlite3.connect(changed_trigger)) as connection:
            connection.executescript(
                """
                DROP TRIGGER events_no_update;
                CREATE TRIGGER events_no_update
                BEFORE UPDATE ON events BEGIN
                    SELECT RAISE(ABORT, 'different event protection');
                END;
                """
            )
            connection.commit()
        before = durable_database_fingerprint(changed_trigger)
        error = io.StringIO()
        with redirect_stderr(error):
            exit_code = main(
                ["--db", str(changed_trigger), "verify", "--json"]
            )
        self.assertEqual(exit_code, 2)
        self.assertIn("changed triggers: events_no_update", error.getvalue())
        self.assertEqual(
            durable_database_fingerprint(changed_trigger),
            before,
        )

        malformed_sessions = (
            Path(self.tempdir.name) / "malformed-sessions.db"
        )
        backup_database(self.database, malformed_sessions)
        connection = sqlite3.connect(malformed_sessions)
        try:
            connection.execute("PRAGMA foreign_keys = OFF")
            connection.execute("DROP TABLE sessions")
            connection.execute(
                "CREATE TABLE sessions(id TEXT PRIMARY KEY)"
            )
            connection.commit()
        finally:
            connection.close()
        before = durable_database_fingerprint(malformed_sessions)
        error = io.StringIO()
        with redirect_stderr(error):
            exit_code = main(
                [
                    "--db",
                    str(malformed_sessions),
                    "session",
                    "list",
                    "--json",
                ]
            )
        self.assertEqual(exit_code, 2)
        self.assertIn(
            "table sessions column definitions differ",
            error.getvalue(),
        )
        self.assertNotIn("OperationalError", error.getvalue())
        self.assertNotIn("Traceback", error.getvalue())
        self.assertEqual(
            durable_database_fingerprint(malformed_sessions),
            before,
        )

    def test_writable_commands_reject_current_schema_drift_without_mutation(
        self,
    ) -> None:
        cases = (
            (
                "missing-trigger",
                "DROP TRIGGER events_no_update",
                "missing triggers: events_no_update",
            ),
            (
                "changed-trigger",
                """
                DROP TRIGGER events_no_update;
                CREATE TRIGGER events_no_update
                BEFORE UPDATE ON events BEGIN
                    SELECT RAISE(ABORT, 'weakened event protection');
                END;
                """,
                "changed triggers: events_no_update",
            ),
            (
                "unexpected-table",
                "CREATE TABLE untrusted_extension(value TEXT)",
                "unexpected tables: untrusted_extension",
            ),
        )
        for label, mutation, expected_error in cases:
            with self.subTest(case=label):
                target = Path(self.tempdir.name) / f"{label}.db"
                backup_database(self.database, target)
                with closing(sqlite3.connect(target)) as connection:
                    connection.executescript(mutation)
                    connection.commit()
                before = durable_database_fingerprint(target)
                error = io.StringIO()
                with redirect_stderr(error):
                    exit_code = main(
                        [
                            "--db",
                            str(target),
                            "learner",
                            "add",
                            f"should-not-exist-{label}",
                            "--json",
                        ]
                    )
                self.assertEqual(exit_code, 2)
                self.assertIn(expected_error, error.getvalue())
                self.assertNotIn("Traceback", error.getvalue())
                self.assertEqual(
                    durable_database_fingerprint(target),
                    before,
                )

    def test_unexpected_sqlite_error_is_reported_without_a_traceback(
        self,
    ) -> None:
        error = io.StringIO()
        with patch(
            "tsq.cli.command_verify",
            side_effect=sqlite3.OperationalError("synthetic read failure"),
        ), redirect_stderr(error):
            exit_code = main(
                ["--db", str(self.database.path), "verify", "--json"]
            )

        self.assertEqual(exit_code, 2)
        self.assertEqual(
            error.getvalue(),
            "error: Database operation failed: synthetic read failure\n",
        )
        self.assertNotIn("OperationalError", error.getvalue())
        self.assertNotIn("Traceback", error.getvalue())


if __name__ == "__main__":
    unittest.main()
