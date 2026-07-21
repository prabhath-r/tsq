# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import io
import json
import tempfile
import unittest
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from tsq.cli import main
from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.errors import ConflictError, ValidationError
from tsq.learner import MODEL_VERSION
from tsq.replay import ProjectionReplay
from tsq.store import Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"
GOLDEN = ROOT / "tests" / "fixtures" / "learner_replay_expected.json"
START = datetime(2100, 4, 5, 9, 0, tzinfo=timezone.utc)


class FixedIds:
    def __init__(self) -> None:
        self.counts: dict[str, int] = defaultdict(int)

    def __call__(self, prefix: str) -> str:
        self.counts[prefix] += 1
        return f"{prefix}_golden_{self.counts[prefix]:03d}"


def build_golden_database(path: Path) -> Database:
    identifiers = FixedIds()
    with (
        patch("tsq.store.new_id", side_effect=identifiers),
        patch("tsq.engine.new_id", side_effect=identifiers),
        patch("tsq.policy.new_id", side_effect=identifiers),
    ):
        database = Database(path)
        database.initialize()
        database.import_corpus(
            *read_and_parse(CORPUS, include_catalog=True)
        )
        engine = AdaptiveEngine(database)
        engine.create_learner("replay-golden", "Projection Replay Fixture")
        session = engine.start_session(
            "replay-golden", "c_clustering", mode="learn", seed=31
        )
        patterns = (
            ("correct", 0.92, 1400, 0, timedelta(minutes=1)),
            ("incorrect", 0.88, 180, 0, timedelta(hours=2, minutes=1)),
            ("uncertain", 0.25, 4200, 1, timedelta(hours=30, minutes=1)),
            ("correct", None, 800, 0, timedelta(hours=54, minutes=1)),
        )
        for index, (answer, confidence, response_ms, hints, answered_after) in enumerate(
            patterns
        ):
            selected_at = START + answered_after - timedelta(minutes=1)
            presentation = engine.next_question(session["id"], now=selected_at)
            if answer == "correct":
                selected_option_id = presentation.question.correct_option.id
            elif answer == "incorrect":
                selected_option_id = next(
                    option.id
                    for option in presentation.question.options
                    if not option.correct
                )
            else:
                selected_option_id = None
            engine.submit_answer(
                presentation.decision_id,
                selected_option_id,
                confidence=confidence,
                response_ms=response_ms,
                hint_count=hints,
                idempotency_key=f"golden-response-{index + 1}",
                now=START + answered_after,
            )
    return database


def golden_projection(report: dict) -> dict:
    return {
        "format_version": report["format_version"],
        "learner_id": report["learner_id"],
        "learner_model_version": report["learner_model_version"],
        "response_count": report["response_count"],
        "final_projection_hash": report["reconstructed_projection_hash"],
        "checkpoints": [
            {
                "revision": checkpoint["revision"],
                "response_event_id": checkpoint["response_event_id"],
                "projection_event_id": checkpoint["projection_event_id"],
                "question_id": checkpoint["question_id"],
                "family_id": checkpoint["family_id"],
                "projection_hash": checkpoint["actual_projection_hash"],
                "state_changes": checkpoint["state_changes"],
            }
            for checkpoint in report["checkpoints"]
        ],
    }


class ProjectionReplayTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = build_golden_database(Path(self.tempdir.name) / "source.db")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def projection_snapshot(self, database: Database) -> dict:
        with database.read() as connection:
            return {
                "revision": connection.execute(
                    "SELECT revision FROM learners WHERE id='replay-golden'"
                ).fetchone()["revision"],
                "skills": [
                    tuple(row)
                    for row in connection.execute(
                        "SELECT * FROM skill_states WHERE learner_id='replay-golden' "
                        "ORDER BY concept_id"
                    )
                ],
                "beliefs": [
                    tuple(row)
                    for row in connection.execute(
                        "SELECT * FROM misconception_beliefs "
                        "WHERE learner_id='replay-golden' ORDER BY misconception_id"
                    )
                ],
                "families": [
                    tuple(row)
                    for row in connection.execute(
                        "SELECT * FROM learner_skill_families "
                        "WHERE learner_id='replay-golden' ORDER BY concept_id, family_id"
                    )
                ],
            }

    def test_replay_matches_every_golden_checkpoint_without_mutating_source(self) -> None:
        before = self.projection_snapshot(self.database)
        report = ProjectionReplay(self.database).check("replay-golden")
        after = self.projection_snapshot(self.database)

        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(before, after)
        self.assertTrue(report["source_projection_matches_replay"])
        self.assertTrue(report["commitment_matches_replay"])
        self.assertTrue(all(item["hash_matches"] for item in report["checkpoints"]))
        self.assertTrue(
            all(item["state_changes_match"] for item in report["checkpoints"])
        )
        expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
        self.assertEqual(golden_projection(report), expected)

    def test_check_detects_projection_tampering_but_replays_committed_history(self) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE skill_states SET mean=mean + 0.75
                   WHERE learner_id='replay-golden' AND concept_id=(
                       SELECT MIN(concept_id) FROM skill_states
                       WHERE learner_id='replay-golden'
                   )"""
            )
        tampered = self.projection_snapshot(self.database)

        report = ProjectionReplay(self.database).check("replay-golden")

        self.assertFalse(report["ok"])
        self.assertFalse(report["source_projection_matches_replay"])
        self.assertTrue(report["commitment_matches_replay"])
        self.assertTrue(all(item["hash_matches"] for item in report["checkpoints"]))
        self.assertIn(
            "learner replay-golden: projection hash mismatch",
            report["recoverable_source_integrity_errors"],
        )
        self.assertEqual(self.projection_snapshot(self.database), tampered)

    def test_rebuild_copy_repairs_projection_and_never_overwrites_target(self) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE skill_states SET variance=variance + 0.2
                   WHERE learner_id='replay-golden' AND concept_id=(
                       SELECT MIN(concept_id) FROM skill_states
                       WHERE learner_id='replay-golden'
                   )"""
            )
        source_tampered = self.projection_snapshot(self.database)
        target = Path(self.tempdir.name) / "rebuilt.db"

        report = ProjectionReplay(self.database).rebuild_copy(
            "replay-golden", target
        )

        self.assertTrue(report["ok"])
        self.assertTrue(report["source_projection_was_repaired"])
        self.assertTrue(target.is_file())
        self.assertEqual(self.projection_snapshot(self.database), source_tampered)
        rebuilt = Database(target)
        self.assertTrue(rebuilt.verify_integrity()["ok"])
        self.assertTrue(ProjectionReplay(rebuilt).check("replay-golden")["ok"])
        with self.assertRaisesRegex(ConflictError, "already exists"):
            ProjectionReplay(self.database).rebuild_copy("replay-golden", target)

    def test_cli_check_is_operational_and_source_schema_is_not_migrated(self) -> None:
        with self.database.transaction() as connection:
            self.database._drop_v6_authoring_triggers(connection)
            connection.execute("UPDATE meta SET value='5' WHERE key='schema_version'")
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                [
                    "--db",
                    str(self.database.path),
                    "replay",
                    "--learner",
                    "replay-golden",
                    "--check",
                    "--json",
                ]
            )
        self.assertEqual(exit_code, 0)
        report = json.loads(output.getvalue())
        self.assertTrue(report["ok"])
        self.assertEqual(report["source_schema_version"], 5)
        with self.database.read() as connection:
            source_version = connection.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()["value"]
        self.assertEqual(source_version, "5")

    def test_unknown_event_or_model_schema_fails_closed(self) -> None:
        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER events_no_update")
            event_id = connection.execute(
                """SELECT event_id FROM events
                   WHERE learner_id='replay-golden'
                     AND event_type='LearnerProjectionAdvanced'
                   ORDER BY stream_version LIMIT 1"""
            ).fetchone()["event_id"]
            connection.execute(
                "UPDATE events SET schema_version=3 WHERE event_id=?", (event_id,)
            )
        with self.assertRaisesRegex(ValidationError, "unsupported schema version 3"):
            ProjectionReplay(self.database).check("replay-golden")

    def test_projection_v2_rejects_malformed_boundary_trace(self) -> None:
        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER events_no_update")
            row = connection.execute(
                """SELECT event_id, payload_json FROM events
                   WHERE learner_id='replay-golden'
                     AND event_type='LearnerProjectionAdvanced'
                   ORDER BY stream_version LIMIT 1"""
            ).fetchone()
            payload = json.loads(row["payload_json"])
            payload["boundary_decision"] = {
                "focus_concept_id": "advanced",
                "selected_concept_id": "foundation",
            }
            connection.execute(
                "UPDATE events SET payload_json=? WHERE event_id=?",
                (json.dumps(payload), row["event_id"]),
            )

        with self.assertRaisesRegex(ValidationError, "boundary decision.*missing"):
            ProjectionReplay(self.database).check("replay-golden")

    def test_replay_uses_historical_family_counts_when_a_family_repeats(self) -> None:
        database = Database(Path(self.tempdir.name) / "repeated-family.db")
        database.initialize()
        database.import_corpus(
            *read_and_parse(CORPUS, include_catalog=True)
        )
        engine = AdaptiveEngine(database)
        engine.create_learner("repeat-family")
        families: list[str] = []
        for index in range(6):
            session = engine.start_session(
                "repeat-family", "c_bias_variance", seed=100 + index
            )
            now = START + timedelta(days=index)
            presentation = engine.next_question(session["id"], now=now)
            families.append(presentation.question.family_id)
            wrong = next(
                option for option in presentation.question.options if not option.correct
            )
            engine.submit_answer(
                presentation.decision_id,
                wrong.id,
                confidence=0.8,
                response_ms=900,
                now=now + timedelta(minutes=1),
            )
            if len(families) != len(set(families)):
                break
        self.assertEqual(len(set(families[:3])), 3, families)
        self.assertLess(len(set(families)), len(families), families)

        report = ProjectionReplay(database).check("repeat-family")

        self.assertTrue(report["ok"], report["errors"])
        self.assertTrue(all(item["hash_matches"] for item in report["checkpoints"]))

    def test_golden_fixture_declares_current_model(self) -> None:
        expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
        self.assertEqual(expected["learner_model_version"], MODEL_VERSION)


if __name__ == "__main__":
    unittest.main()
