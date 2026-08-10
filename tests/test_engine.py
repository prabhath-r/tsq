# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from tsq.authoring import deterministic_test_pipeline
from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.errors import ConflictError, ExhaustedError, ValidationError
from tsq.models import SessionPhase
from tsq.store import SCHEMA_VERSION, Database

from tests.schema_upgrade_helpers import (
    durable_database_fingerprint,
    restore_pre_shadow_schema,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
START = datetime(2020, 6, 7, 9, 0, tzinfo=timezone.utc)


class EngineTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.tempdir.name) / "test.db")
        self.database.initialize()
        self.database.import_corpus(
            *read_and_parse(CORPUS, include_catalog=True)
        )
        self.engine = AdaptiveEngine(self.database)
        self.engine.create_learner("learner-1", "Test Learner")
        self.test_clock = START

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def start(self, *, mode: str = "learn", seed: int = 17):
        return self.engine.start_session(
            "learner-1",
            "t_machine_learning",
            mode=mode,
            seed=seed,
            now=self.test_clock,
        )

    def next_at(self, session_id: str):
        self.test_clock += timedelta(minutes=1)
        return self.engine.next_question(
            session_id, now=self.test_clock
        )

    def submit_at(self, decision_id: str, selected_option_id, **kwargs):
        self.test_clock += timedelta(minutes=1)
        return self.engine.submit_answer(
            decision_id,
            selected_option_id,
            now=self.test_clock,
            **kwargs,
        )

    def test_session_start_accepts_an_explicit_utc_clock(self) -> None:
        local_time = datetime(
            2099,
            3,
            4,
            12,
            30,
            tzinfo=timezone(timedelta(hours=5, minutes=30)),
        )
        expected = local_time.astimezone(timezone.utc).isoformat()

        session = self.engine.start_session(
            "learner-1",
            "c_attention",
            seed=91,
            idempotency_key="explicit-session-clock",
            now=local_time,
        )

        self.assertEqual(session["created_at"], expected)
        self.assertEqual(session["updated_at"], expected)
        with self.database.read() as connection:
            event = connection.execute(
                """SELECT occurred_at FROM events
                   WHERE idempotency_key='explicit-session-clock'"""
            ).fetchone()
        self.assertEqual(event["occurred_at"], expected)

        retried = self.engine.start_session(
            "learner-1",
            "c_attention",
            seed=91,
            idempotency_key="explicit-session-clock",
            now=local_time + timedelta(days=1),
        )
        self.assertEqual(retried["id"], session["id"])
        self.assertEqual(retried["created_at"], expected)

    def test_session_start_rejects_a_naive_explicit_clock(self) -> None:
        with self.assertRaisesRegex(ValidationError, "timezone-aware"):
            self.engine.start_session(
                "learner-1",
                "c_attention",
                seed=91,
                now=datetime(2099, 3, 4, 12, 30),
            )

    def downgrade_current_database_to_v3(self) -> None:
        """Strip v4 additions while retaining populated v3 learner/event data."""
        with self.database.transaction() as connection:
            restore_pre_shadow_schema(connection)
            self.database._drop_corpus_registry_triggers(connection)
            self.database._drop_release_snapshot_triggers(connection)
            for trigger in (
                "events_no_update",
                "events_no_delete",
                "attempts_validate_insert",
                "attempts_no_update",
                "attempts_no_delete",
                "learning_artifacts_no_update",
                "learning_artifacts_no_delete",
                "learning_actions_validate_insert",
                "learning_actions_no_update",
                "learning_actions_no_delete",
            ):
                connection.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            connection.execute("DROP INDEX IF EXISTS idx_one_pending_decision")

            # These tables were introduced after the legacy schema being
            # reconstructed and must not leak into the migration fixture.
            performance_tables = (
                "shadow_evidence_bundles",
                "task_evaluations",
                "performance_actions",
                "performance_scoring_claims",
                "performance_attempts",
                "release_performance_tasks",
                "performance_task_releases",
                "performance_tasks",
            )
            performance_triggers = connection.execute(
                """SELECT name FROM sqlite_master
                   WHERE type='trigger' AND (
                       name='events_respect_performance_scoring_claim'
                       OR tbl_name IN (
                           'shadow_evidence_bundles', 'task_evaluations',
                           'performance_actions', 'performance_scoring_claims',
                           'performance_attempts',
                           'release_performance_tasks',
                           'performance_task_releases', 'performance_tasks'
                       )
                   )"""
            ).fetchall()
            for trigger in performance_triggers:
                escaped = trigger["name"].replace('"', '""')
                connection.execute(f'DROP TRIGGER "{escaped}"')
            for table in performance_tables:
                connection.execute(f"DROP TABLE {table}")
            connection.execute("DROP TABLE learning_actions")
            connection.execute("DROP TABLE learning_artifacts")
            connection.execute("DROP TABLE learner_objective_families")
            connection.execute("DROP TABLE objective_states")
            connection.execute("DROP TABLE release_option_objectives")
            connection.execute("DROP TABLE release_question_objectives")
            connection.execute("DROP TABLE release_objective_edges")
            connection.execute("DROP TABLE release_objective_graphs")
            connection.execute("DROP TABLE release_learning_objectives")

            for column in ("command_hash", "outcome_json"):
                connection.execute(f"ALTER TABLE attempts DROP COLUMN {column}")
            for column in (
                "question_version",
                "question_content_hash",
                "question_status",
                "evidence_weight",
                "corpus_release_id",
                "session_revision",
                "learner_revision",
                "focus_concept_id",
                "focus_misconception_id",
                "question_objective_id",
                "focus_objective_id",
                "pedagogical_role",
                "focus_valid",
                "invalidated_at",
                "invalidation_reason",
            ):
                connection.execute(f"ALTER TABLE decisions DROP COLUMN {column}")
            connection.execute(
                "ALTER TABLE sessions DROP COLUMN remediation_path_json"
            )
            connection.execute("ALTER TABLE sessions DROP COLUMN focus_objective_id")
            connection.execute("ALTER TABLE sessions DROP COLUMN topic_id")
            connection.execute("ALTER TABLE sessions DROP COLUMN exploration_mode")
            for table, column in (
                ("sessions", "corpus_release_id"),
                ("sessions", "revision"),
                ("learners", "revision"),
                ("concepts", "content_hash"),
                ("misconceptions", "content_hash"),
                ("sources", "content_hash"),
            ):
                connection.execute(f"ALTER TABLE {table} DROP COLUMN {column}")

            connection.execute("DROP TABLE learning_objectives")

            connection.execute("DROP TABLE release_question_topics")
            connection.execute("DROP TABLE release_topic_concepts")
            connection.execute("DROP TABLE release_topics")
            connection.execute("DROP TABLE release_domains")
            connection.execute("DROP TABLE release_questions")
            connection.execute("DROP TABLE release_edges")
            connection.execute("DROP TABLE release_concepts")
            connection.execute("DROP TABLE release_misconceptions")
            connection.execute("DROP TABLE release_sources")
            connection.execute("DROP TABLE corpus_releases")
            connection.execute("DROP TABLE stream_heads")
            connection.execute("DELETE FROM meta WHERE key = 'active_corpus_release'")

            for event in connection.execute(
                "SELECT event_id, event_type, payload_json, metadata_json FROM events"
            ).fetchall():
                payload = json.loads(event["payload_json"])
                metadata = json.loads(event["metadata_json"])
                payload.pop("corpus_release_id", None)
                metadata.pop("corpus_release_id", None)
                if event["event_type"] == "SessionStarted":
                    payload.pop("topic_id", None)
                    payload.pop("exploration_mode", None)
                    payload.pop("requested_root_concept_id", None)
                connection.execute(
                    """UPDATE events SET payload_json = ?, metadata_json = ?
                       WHERE event_id = ?""",
                    (
                        json.dumps(payload, sort_keys=True, separators=(",", ":")),
                        json.dumps(metadata, sort_keys=True, separators=(",", ":")),
                        event["event_id"],
                    ),
                )

            streams = connection.execute(
                "SELECT DISTINCT stream_id FROM events ORDER BY stream_id"
            ).fetchall()
            for stream in streams:
                previous_hash = None
                events = connection.execute(
                    "SELECT * FROM events WHERE stream_id = ? ORDER BY stream_version",
                    (stream["stream_id"],),
                ).fetchall()
                for event in events:
                    material = "|".join(
                        [
                            previous_hash or "",
                            event["stream_id"],
                            str(event["stream_version"]),
                            event["event_type"],
                            event["payload_json"],
                            event["metadata_json"],
                        ]
                    ).encode("utf-8")
                    payload_hash = hashlib.sha256(material).hexdigest()
                    connection.execute(
                        "UPDATE events SET previous_hash = ?, payload_hash = ? WHERE event_id = ?",
                        (previous_hash, payload_hash, event["event_id"]),
                    )
                    previous_hash = payload_hash
            connection.execute("UPDATE meta SET value = '3' WHERE key = 'schema_version'")

    def test_schema_version_is_committed(self) -> None:
        with self.database.read() as connection:
            value = connection.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()["value"]
        self.assertEqual(SCHEMA_VERSION, 23)
        self.assertEqual(value, str(SCHEMA_VERSION))

    def test_topic_session_preserves_continuity_then_explores_explicitly(self) -> None:
        session = self.engine.start_session(
            "learner-1",
            "Transformers",
            mode="learn",
            seed=9,
            now=self.test_clock,
        )
        self.assertEqual(session["topic_id"], "t_transformers")
        self.assertEqual(session["exploration_mode"], "adaptive")
        owned = self.database.topic_owned_concepts("t_transformers")

        for index in range(3):
            presentation = self.next_at(session["id"])
            decision = self.database.recent_decisions(session["id"], 1)[0]
            self.assertEqual(decision["pedagogical_role"], "main")
            self.assertIn(presentation.question.primary_concept_id, owned)
            self.submit_at(
                presentation.decision_id,
                presentation.question.correct_option.id,
                confidence=0.9,
                response_ms=1800,
                idempotency_key=f"topic-continuity-{index}",
            )

        exploration = self.next_at(session["id"])
        decision = self.database.recent_decisions(session["id"], 1)[0]
        self.assertEqual(decision["pedagogical_role"], "exploration_probe")
        self.assertNotIn(exploration.question.primary_concept_id, owned)
        self.assertIn("deliberate_related_topic_probe", exploration.rationale)

        wrong = next(option for option in exploration.question.options if not option.correct)
        result = self.submit_at(
            exploration.decision_id,
            wrong.id,
            confidence=0.9,
            response_ms=1800,
            idempotency_key="topic-exploration-wrong",
        )
        self.assertEqual(result.next_phase, SessionPhase.REMEDIATE)
        self.assertEqual(
            result.focus_concept_id, exploration.question.primary_concept_id
        )
        repair = self.next_at(session["id"])
        repair_decision = self.database.recent_decisions(session["id"], 1)[0]
        self.assertEqual(repair_decision["pedagogical_role"], "remediation_probe")
        self.assertNotEqual(repair.question.family_id, exploration.question.family_id)
        self.assertIn(result.focus_misconception_id, repair.question.misconception_ids)

    def test_unserviceable_optional_exploration_falls_back_to_requested_topic(
        self,
    ) -> None:
        session = self.engine.start_session(
            "learner-1",
            "Transformers",
            mode="learn",
            seed=109,
            now=self.test_clock,
        )
        release_id = session["corpus_release_id"]
        base_scope = self.database.topic_scope("t_transformers", release_id)
        catalog = self.database.get_catalog(release_id)
        topic = next(
            item for item in catalog["topics"] if item["id"] == "t_transformers"
        )
        exploration_scope: set[str] = set()
        for related_topic_id in topic["related_topic_ids"]:
            exploration_scope.update(
                self.database.topic_scope(related_topic_id, release_id)
            )
        exploration_scope -= base_scope
        self.assertTrue(exploration_scope)

        original_questions_for_scope = self.database.questions_for_scope

        def omit_exploration_scope(scope: set[str], **kwargs):
            if set(scope) == exploration_scope:
                return []
            return original_questions_for_scope(scope, **kwargs)

        with (
            patch.object(
                self.engine.policy,
                "_should_explore",
                return_value=True,
            ),
            patch.object(
                self.database,
                "questions_for_scope",
                side_effect=omit_exploration_scope,
            ),
        ):
            presentation = self.next_at(session["id"])

        decision = self.database.recent_decisions(session["id"], 1)[0]
        self.assertEqual(decision["pedagogical_role"], "main")
        self.assertIn(
            presentation.question.primary_concept_id,
            self.database.topic_owned_concepts("t_transformers", release_id),
        )
        self.assertIn(
            "exploration_unserviceable=no_approved_questions:",
            presentation.rationale,
        )
        with self.database.read() as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) AS n FROM decisions WHERE session_id = ?",
                    (session["id"],),
                ).fetchone()["n"],
                1,
            )
            self.assertEqual(
                connection.execute(
                    """SELECT COUNT(*) AS n FROM events
                       WHERE session_id = ?
                         AND event_type = 'CorpusGapDetected'""",
                    (session["id"],),
                ).fetchone()["n"],
                0,
            )
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_session_report_exposes_time_difficulty_and_uncertainty_paths(self) -> None:
        session = self.engine.start_session(
            "learner-1",
            "LLM Agents",
            mode="learn",
            seed=21,
            now=self.test_clock,
        )
        first = self.next_at(session["id"])
        self.submit_at(
            first.decision_id,
            first.question.correct_option.id,
            confidence=0.85,
            response_ms=2400,
            idempotency_key="report-first",
        )
        second = self.next_at(session["id"])
        self.submit_at(
            second.decision_id,
            None,
            confidence=0.2,
            response_ms=5100,
            idempotency_key="report-second",
        )
        self.engine.end_session(
            session["id"],
            status="completed",
            reason="report_test",
            now=self.test_clock + timedelta(seconds=1),
        )

        report = self.engine.session_report(session["id"])

        self.assertEqual(report["topic"]["id"], "t_llm_agents")
        self.assertEqual(report["questions_answered"], 2)
        self.assertEqual(report["abstained"], 1)
        self.assertEqual(report["response_time"]["active_seconds"], 7.5)
        self.assertEqual(
            report["response_time"]["selection_window_inconsistencies"], 0
        )
        self.assertFalse(report["response_time"]["active_exceeds_session_wall"])
        self.assertIn(
            "authoritative selection-event window",
            report["response_time"]["evidence_contract"],
        )
        self.assertIn("cross_topic_definition", report)
        self.assertIn("outside_requested_topic_questions", report)
        self.assertEqual(report["behavior_trace"]["submitted_hint_count"], 0)
        self.assertIsNotNone(report["difficulty"]["average"])
        self.assertEqual(report["unique_families"], 2)
        self.assertTrue(
            report["concept_changes"] or report["objective_changes"]
        )
        self.assertIn("scale", report["difficulty"])
        self.assertEqual(len(report["adaptive_path"]), 2)
        self.assertTrue(
            all("transition_reason" in step for step in report["adaptive_path"])
        )
        self.assertTrue(report["concept_performance"])
        self.assertTrue(report["diagnostic_findings"])
        self.assertTrue(
            any(
                "uncertain_or_noncredible_evidence"
                in finding["attention_reasons"]
                for finding in report["diagnostic_findings"]
            )
        )
        self.assertIn("boundary_algorithm_version", report)
        self.assertIn("transition_counts", report["adaptive_routing"])
        self.assertEqual(
            sum(report["adaptive_routing"]["transition_counts"].values()), 2
        )

        profile = self.engine.profile("learner-1", root_concept_id="LLM Agents")
        self.assertIn("boundary_algorithm_version", profile)
        assessed = [
            skill for skill in profile["skills"] if skill["evidence_mass"] > 0
        ]
        self.assertTrue(assessed)
        self.assertTrue(
            all(
                {
                    "intrinsic_readiness",
                    "prerequisite_support",
                    "effective_readiness",
                    "bottleneck_concept_id",
                }
                <= set(skill)
                for skill in assessed
            )
        )

    def test_v1_database_is_migrated_and_question_hashes_are_backfilled(self) -> None:
        with self.database.transaction() as connection:
            restore_pre_shadow_schema(connection)
            self.database._drop_corpus_registry_triggers(connection)
            connection.execute("ALTER TABLE questions DROP COLUMN content_hash")
            connection.execute(
                "ALTER TABLE learner_skill_families DROP COLUMN delayed_unguided_correct_at"
            )
            connection.execute("ALTER TABLE learner_skill_families DROP COLUMN kind")
            connection.execute("UPDATE meta SET value = '1' WHERE key = 'schema_version'")
        self.database.initialize()
        with self.database.read() as connection:
            version = connection.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()["value"]
            question_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(questions)")
            }
            family_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(learner_skill_families)")
            }
            missing_hashes = connection.execute(
                "SELECT COUNT(*) AS n FROM questions WHERE content_hash IS NULL"
            ).fetchone()["n"]
        self.assertEqual(version, str(SCHEMA_VERSION))
        self.assertIn("content_hash", question_columns)
        self.assertIn("delayed_unguided_correct_at", family_columns)
        self.assertIn("kind", family_columns)
        self.assertEqual(missing_hashes, 0)

    def test_populated_v3_database_requires_v22_intermediate(self) -> None:
        session = self.start(seed=29)
        presentation = self.next_at(session["id"])
        self.submit_at(
            presentation.decision_id,
            presentation.question.correct_option.id,
            confidence=0.72,
            response_ms=840,
            idempotency_key="migration-v3-answer",
        )
        self.downgrade_current_database_to_v3()
        before = durable_database_fingerprint(self.database.path)

        with self.assertRaisesRegex(
            ConflictError,
            (
                "Schema v3 contains learner response history from before "
                "the durable evidence-integrity boundary.*schema v22.*"
                "TSQ v0.1.0"
            ),
        ):
            self.database.initialize()

        self.assertEqual(durable_database_fingerprint(self.database.path), before)
        with self.database.read() as connection:
            version = connection.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()["value"]
            attempt = connection.execute(
                "SELECT id FROM attempts WHERE decision_id = ?",
                (presentation.decision_id,),
            ).fetchone()
        self.assertEqual(version, "3")
        self.assertIsNotNone(attempt)

    def test_incomplete_v4_release_history_fails_closed(self) -> None:
        with self.database.transaction() as connection:
            restore_pre_shadow_schema(connection)
            self.database._drop_release_snapshot_triggers(connection)
            connection.execute("DROP TABLE release_sources")
            connection.execute(
                "UPDATE meta SET value = '4' WHERE key = 'schema_version'"
            )
        with self.assertRaisesRegex(
            ConflictError, "cannot be reconstructed safely"
        ):
            self.database.initialize()

    def test_pending_decision_is_stable(self) -> None:
        session = self.start()
        first = self.engine.next_question(session["id"])
        second = self.engine.next_question(session["id"])
        self.assertEqual(first.decision_id, second.decision_id)
        self.assertEqual(first.option_order, second.option_order)

    def test_concurrent_next_calls_persist_exactly_one_pending_decision(self) -> None:
        session = self.start(seed=97)
        with ThreadPoolExecutor(max_workers=8) as executor:
            presentations = list(
                executor.map(
                    lambda _: self.engine.next_question(session["id"]),
                    range(8),
                )
            )
        self.assertEqual(
            {presentation.decision_id for presentation in presentations},
            {presentations[0].decision_id},
        )
        with self.database.read() as connection:
            decisions = connection.execute(
                """SELECT COUNT(*) AS n FROM decisions
                   WHERE session_id = ? AND consumed_at IS NULL
                     AND invalidated_at IS NULL""",
                (session["id"],),
            ).fetchone()["n"]
            events = connection.execute(
                """SELECT COUNT(*) AS n FROM events
                   WHERE session_id = ? AND event_type = 'QuestionSelected'""",
                (session["id"],),
            ).fetchone()["n"]
        self.assertEqual(decisions, 1)
        self.assertEqual(events, 1)

    def test_concurrent_cross_session_answers_apply_one_learner_revision(self) -> None:
        first_session = self.start(seed=137)
        second_session = self.start(seed=139)
        presentations = (
            self.next_at(first_session["id"]),
            self.next_at(second_session["id"]),
        )
        answer_time = self.test_clock + timedelta(minutes=1)

        def submit(index: int):
            presentation = presentations[index]
            try:
                result = self.engine.submit_answer(
                    presentation.decision_id,
                    presentation.question.correct_option.id,
                    confidence=0.9,
                    response_ms=900,
                    idempotency_key=f"concurrent-cross-session-{index}",
                    now=answer_time,
                )
                return "applied", result.interaction_id
            except ConflictError as exc:
                return "stale", str(exc)

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(submit, range(2)))
        self.test_clock = answer_time
        self.assertEqual([outcome[0] for outcome in outcomes].count("applied"), 1)
        self.assertEqual([outcome[0] for outcome in outcomes].count("stale"), 1)
        stale_index = next(
            index for index, outcome in enumerate(outcomes) if outcome[0] == "stale"
        )
        replacement = self.engine.next_question(
            (first_session, second_session)[stale_index]["id"]
        )
        self.assertNotEqual(
            replacement.decision_id, presentations[stale_index].decision_id
        )
        report = self.database.verify_integrity()
        self.assertTrue(report["ok"], report["errors"])

    def test_presented_but_unanswered_question_counts_as_exposure(self) -> None:
        session = self.start(seed=101)
        presentation = self.engine.next_question(session["id"])
        exposure = self.database.get_exposure_summary(
            "learner-1",
            question_ids={presentation.question.id},
            family_ids={presentation.question.family_id},
        )
        self.assertEqual(exposure["questions"][presentation.question.id], 1)
        self.assertEqual(
            exposure["families"][presentation.question.family_id]["count"], 1
        )

    def test_recent_family_is_reserved_across_sessions_before_answer(self) -> None:
        selected_families: set[str] = set()
        selection_time = datetime(2100, 1, 1, tzinfo=timezone.utc)
        for offset in range(4):
            session = self.engine.start_session(
                "learner-1",
                "Transformers",
                mode="review" if offset % 2 else "learn",
                seed=301 + offset,
            )
            presentation = self.engine.next_question(
                session["id"],
                now=selection_time + timedelta(seconds=offset),
            )
            self.assertNotIn(
                presentation.question.family_id,
                selected_families,
            )
            selected_families.add(presentation.question.family_id)

        self.assertEqual(len(selected_families), 4)

    def test_parallel_session_stale_decision_is_rejected_and_invalidated(self) -> None:
        first_session = self.start(seed=103)
        second_session = self.start(seed=107)
        first = self.next_at(first_session["id"])
        stale = self.next_at(second_session["id"])

        self.submit_at(
            first.decision_id,
            first.question.correct_option.id,
            confidence=0.8,
            response_ms=900,
            idempotency_key="advance-parallel-learner",
        )
        with self.assertRaises(ConflictError):
            self.engine.submit_answer(
                stale.decision_id,
                stale.question.correct_option.id,
                idempotency_key="reject-stale-parallel-answer",
            )

        replacement = self.next_at(second_session["id"])
        repeated = self.engine.next_question(second_session["id"])
        self.assertNotEqual(replacement.decision_id, stale.decision_id)
        self.assertEqual(repeated.decision_id, replacement.decision_id)
        with self.database.read() as connection:
            stale_row = connection.execute(
                "SELECT * FROM decisions WHERE id = ?", (stale.decision_id,)
            ).fetchone()
            invalidation_events = connection.execute(
                """SELECT COUNT(*) AS n FROM events
                   WHERE event_type = 'DecisionInvalidated'
                     AND causation_id = ?""",
                (stale.decision_id,),
            ).fetchone()["n"]
        self.assertIsNotNone(stale_row["invalidated_at"])
        self.assertEqual(
            stale_row["invalidation_reason"], "learner_projection_advanced"
        )
        self.assertEqual(invalidation_events, 1)
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_emergency_revocation_overrides_pinned_release_and_pending_choice(self) -> None:
        session = self.start(seed=131)
        revoked_presentation = self.engine.next_question(session["id"])
        first = self.database.revoke_question(
            revoked_presentation.question.id,
            "Answer key is under emergency review.",
            idempotency_key="revoke-pending-question",
        )
        replay = self.database.revoke_question(
            revoked_presentation.question.id,
            "Answer key is under emergency review.",
            idempotency_key="revoke-pending-question",
        )
        self.assertFalse(first["idempotent"])
        self.assertTrue(replay["idempotent"])
        self.assertIsNone(self.database.pending_presentation(session["id"]))
        with self.assertRaisesRegex(ConflictError, "emergency-revoked"):
            self.engine.submit_answer(
                revoked_presentation.decision_id,
                revoked_presentation.question.correct_option.id,
                idempotency_key="answer-revoked-question",
            )

        replacement = self.engine.next_question(session["id"])
        self.assertNotEqual(
            replacement.question.id, revoked_presentation.question.id
        )
        with self.database.read() as connection:
            invalidated = connection.execute(
                "SELECT * FROM decisions WHERE id = ?",
                (revoked_presentation.decision_id,),
            ).fetchone()
            safety_events = connection.execute(
                """SELECT COUNT(*) AS n FROM events
                   WHERE event_type = 'QuestionEmergencyRevoked'"""
            ).fetchone()["n"]
        self.assertEqual(
            invalidated["invalidation_reason"], "question_emergency_revoked"
        )
        self.assertIsNotNone(invalidated["invalidated_at"])
        self.assertEqual(safety_events, 1)
        with self.assertRaises(sqlite3.IntegrityError):
            with self.database.transaction() as connection:
                connection.execute(
                    """UPDATE question_revocations SET reason = 'rewritten'
                       WHERE question_id = ?""",
                    (revoked_presentation.question.id,),
                )
        report = self.database.verify_integrity()
        self.assertTrue(report["ok"], report["errors"])

    def test_revocation_integrity_uses_recorded_order_not_domain_timestamps(self) -> None:
        future = datetime(2100, 1, 1, tzinfo=timezone.utc)
        session = self.start(seed=137)
        presentation = self.engine.next_question(session["id"], now=future)
        self.engine.submit_answer(
            presentation.decision_id,
            presentation.question.correct_option.id,
            confidence=0.8,
            response_ms=900,
            idempotency_key="future-domain-time-answer",
            now=future + timedelta(minutes=1),
        )

        self.database.revoke_question(
            presentation.question.id,
            "Post-use safety review.",
            idempotency_key="revoke-after-future-domain-times",
        )

        report = self.database.verify_integrity()
        self.assertTrue(report["ok"], report["errors"])

    def test_revocation_integrity_detects_operations_recorded_after_revocation(self) -> None:
        session = self.start(seed=139)
        presentation = self.engine.next_question(session["id"])
        self.database.revoke_question(
            presentation.question.id,
            "Unsafe item discovered before response.",
            idempotency_key="revoke-before-forged-operations",
        )

        accepted_at = datetime.now(timezone.utc)
        selected_option_id = presentation.question.correct_option.id
        with self.database.transaction() as connection:
            decision = connection.execute(
                "SELECT * FROM decisions WHERE id = ?",
                (presentation.decision_id,),
            ).fetchone()
            original_selection = connection.execute(
                """SELECT * FROM events
                   WHERE event_type = 'QuestionSelected'
                     AND json_extract(payload_json, '$.decision_id') = ?""",
                (presentation.decision_id,),
            ).fetchone()
            self.database.append_event(
                connection,
                stream_id=f"learner:{decision['learner_id']}",
                event_type="QuestionSelected",
                payload=json.loads(original_selection["payload_json"]),
                metadata=json.loads(original_selection["metadata_json"]),
                learner_id=decision["learner_id"],
                session_id=decision["session_id"],
                occurred_at=accepted_at,
            )

            command_payload = {
                "decision_id": decision["id"],
                "question_id": decision["question_id"],
                "question_version": decision["question_version"],
                "selected_option_id": selected_option_id,
                "is_correct": True,
                "confidence": 0.8,
                "response_ms": 900,
                "hint_count": 0,
                "feedback_shown": True,
                "presented_order": list(presentation.option_order),
            }
            response = self.database.append_event(
                connection,
                stream_id=f"learner:{decision['learner_id']}",
                event_type="ResponseSubmitted",
                payload=command_payload,
                metadata={
                    "policy_version": decision["policy_version"],
                    "corpus_release_id": decision["corpus_release_id"],
                    "question_content_hash": decision["question_content_hash"],
                    "question_status": decision["question_status"],
                    "evidence_weight": decision["evidence_weight"],
                    "selection_learner_revision": decision["learner_revision"],
                },
                learner_id=decision["learner_id"],
                session_id=decision["session_id"],
                causation_id=decision["id"],
                occurred_at=accepted_at,
            )
            command_hash = hashlib.sha256(
                json.dumps(
                    command_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            connection.execute(
                """INSERT INTO attempts(
                       id, event_id, decision_id, session_id, learner_id,
                       question_id, question_version, family_id,
                       presented_order_json, selected_option_id, is_correct,
                       confidence, response_ms, hint_count, feedback_shown,
                       answered_at, command_hash, outcome_json
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, 0, 1, ?, ?, NULL)""",
                (
                    "att_forged_after_revocation",
                    response["event_id"],
                    decision["id"],
                    decision["session_id"],
                    decision["learner_id"],
                    decision["question_id"],
                    decision["question_version"],
                    presentation.question.family_id,
                    decision["option_order_json"],
                    selected_option_id,
                    0.8,
                    900,
                    accepted_at.isoformat(),
                    command_hash,
                ),
            )
            connection.execute(
                "UPDATE decisions SET consumed_at = ? WHERE id = ?",
                (accepted_at.isoformat(), decision["id"]),
            )

        report = self.database.verify_integrity()
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("decisions were selected after revocation" in error for error in report["errors"]),
            report["errors"],
        )
        self.assertTrue(
            any("attempts were accepted after revocation" in error for error in report["errors"]),
            report["errors"],
        )

    def test_live_corpus_gap_creates_one_durable_authoring_demand(self) -> None:
        verification_kinds = {
            "application",
            "calculation",
            "comparison",
            "counterfactual",
            "debugging",
            "transfer",
        }
        verification_questions = [
            question
            for question in self.database.questions_for_scope(
                {"c_data_leakage"},
                release_id=self.database.get_active_release_id(),
                limit=600,
            )
            if question.primary_concept_id == "c_data_leakage"
            and question.kind.value in verification_kinds
        ]
        self.assertGreaterEqual(len(verification_questions), 1)
        for question in verification_questions:
            self.database.revoke_question(
                question.id,
                "Gap regression: remove every live verification family.",
            )
        session = self.engine.start_session(
            "learner-1", "c_data_leakage", mode="learn", seed=113
        )
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE sessions SET phase = 'verify',
                          focus_concept_id = 'c_data_leakage'
                   WHERE id = ?""",
                (session["id"],),
            )
        for _ in range(2):
            with self.assertRaisesRegex(ExhaustedError, r"^Corpus gap:"):
                self.engine.next_question(session["id"])
        with self.database.read() as connection:
            events = connection.execute(
                """SELECT payload_json FROM events
                   WHERE event_type = 'CorpusGapDetected'"""
            ).fetchall()
            jobs = connection.execute(
                "SELECT blueprint_json, status FROM generation_jobs"
            ).fetchall()
        self.assertEqual(len(events), 1)
        self.assertEqual(len(jobs), 1)
        payload = json.loads(events[0]["payload_json"])
        blueprint = json.loads(jobs[0]["blueprint_json"])
        self.assertEqual(payload["job_ids"], [payload["job_id"]])
        self.assertEqual(blueprint["concept_id"], "c_data_leakage")
        self.assertEqual(blueprint["kind"], "transfer")
        self.assertTrue(blueprint["source_ids"])
        self.assertEqual(
            blueprint["corpus_release_id"],
            self.database.get_active_release_id(),
        )
        self.assertIsNone(blueprint["learning_objective_id"])
        self.assertEqual(blueprint["coverage_goal"], "live_corpus_gap")
        self.assertEqual(jobs[0]["status"], "planned")

    def test_topic_corpus_gap_targets_an_owned_objective(self) -> None:
        session = self.engine.start_session(
            "learner-1",
            "Transformers",
            mode="learn",
            seed=127,
        )
        self.engine._record_corpus_gap(
            session["id"],
            message="Corpus gap: synthetic topic-bound regression.",
            now=datetime.now(timezone.utc),
        )
        with self.database.read() as connection:
            row = connection.execute(
                "SELECT id, blueprint_json FROM generation_jobs"
            ).fetchone()
        self.assertIsNotNone(row)
        blueprint = json.loads(row["blueprint_json"])
        self.assertIn(
            blueprint["concept_id"],
            self.database.topic_owned_concepts("t_transformers"),
        )
        objective = next(
            objective
            for objective in self.database.get_learning_objectives()
            if objective.id == blueprint["learning_objective_id"]
        )
        self.assertIn(blueprint["concept_id"], objective.concept_ids)
        self.assertEqual(
            blueprint["learning_objective_operation"],
            objective.operation.value,
        )
        self.assertEqual(blueprint["coverage_goal"], "live_corpus_gap")

    def test_review_with_only_cross_owner_items_due_is_not_a_corpus_gap(
        self,
    ) -> None:
        now = datetime(2100, 5, 1, 9, 0, tzinfo=timezone.utc)
        session = self.engine.start_session(
            "learner-1",
            "Transformers",
            mode="review",
            seed=131,
            now=now,
        )
        release_id = session["corpus_release_id"]
        owned = self.database.topic_owned_concepts("t_transformers")

        def cross_owner_only_exposure(
            learner_id: str,
            *,
            question_ids: set[str] | None = None,
            family_ids: set[str] | None = None,
        ) -> dict[str, object]:
            self.assertEqual(learner_id, "learner-1")
            families: dict[str, dict[str, object]] = {}
            for question_id in question_ids or ():
                question = self.database.get_question(
                    question_id,
                    release_id=release_id,
                )
                owner = (
                    question.objective.primary_concept_id
                    if question.objective is not None
                    else question.primary_concept_id
                )
                if owner in owned:
                    families[question.family_id] = {
                        "count": 1,
                        "last_at": (now - timedelta(minutes=1)).isoformat(),
                    }
            return {"questions": {}, "families": families}

        with patch.object(
            self.database,
            "get_exposure_summary",
            side_effect=cross_owner_only_exposure,
        ):
            with self.assertRaisesRegex(
                ExhaustedError,
                r"due yet for the requested review target",
            ):
                self.engine.next_question(session["id"], now=now)

        with self.database.read() as connection:
            gap_events = connection.execute(
                """SELECT COUNT(*) AS n FROM events
                   WHERE learner_id = 'learner-1'
                     AND event_type = 'CorpusGapDetected'"""
            ).fetchone()["n"]
            jobs = connection.execute(
                "SELECT COUNT(*) AS n FROM generation_jobs"
            ).fetchone()["n"]
        self.assertEqual(gap_events, 0)
        self.assertEqual(jobs, 0)
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_focused_live_gap_preserves_exact_objective_and_misconception(self) -> None:
        objective = next(
            objective
            for objective in self.database.get_learning_objectives()
            if objective.id == "lo_causal_visibility"
        )
        with self.database.read() as connection:
            misconception_id = connection.execute(
                """SELECT option.misconception_id
                   FROM release_option_objectives mapping
                   JOIN options option
                     ON option.question_id = mapping.question_id
                    AND option.option_id = mapping.option_id
                   JOIN release_questions membership
                     ON membership.release_id = mapping.release_id
                    AND membership.question_id = mapping.question_id
                   WHERE mapping.release_id = ?
                     AND mapping.objective_id = ?
                     AND membership.status IN ('approved', 'calibrated')
                     AND option.is_correct = 0
                     AND NOT EXISTS (
                         SELECT 1
                         FROM question_revocations revoked
                         WHERE revoked.question_id = mapping.question_id
                     )
                     AND option.misconception_id IS NOT NULL
                   ORDER BY option.misconception_id LIMIT 1""",
                (
                    self.database.get_active_release_id(connection),
                    objective.id,
                ),
            ).fetchone()["misconception_id"]
        session = self.engine.start_session(
            "learner-1", "Transformers", mode="learn", seed=191
        )
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE sessions
                   SET phase = 'verify', focus_concept_id = ?,
                       focus_misconception_id = ?, focus_objective_id = ?
                   WHERE id = ?""",
                (
                    objective.primary_concept_id,
                    misconception_id,
                    objective.id,
                    session["id"],
                ),
            )

        self.engine._record_corpus_gap(
            session["id"],
            message="Corpus gap: exact objective regression.",
            now=datetime.now(timezone.utc),
        )

        with self.database.read() as connection:
            row = connection.execute(
                "SELECT id, blueprint_json FROM generation_jobs"
            ).fetchone()
            event = connection.execute(
                """SELECT payload_json FROM events
                   WHERE event_type = 'CorpusGapDetected'"""
            ).fetchone()
        blueprint = json.loads(row["blueprint_json"])
        payload = json.loads(event["payload_json"])
        self.assertEqual(blueprint["learning_objective_id"], objective.id)
        self.assertEqual(
            blueprint["target_misconception_id"], misconception_id
        )
        self.assertEqual(blueprint["misconception_ids"], [misconception_id])
        self.assertEqual(payload["focus_objective_id"], objective.id)
        generated = deterministic_test_pipeline(self.database).run_job(
            row["id"], "Pinned primary-source context for the exact live gap."
        )
        self.assertEqual(generated["status"], "reviewed")
        self.assertEqual(generated["item"]["status"], "approved")
        self.assertEqual(
            generated["item"]["learning_objective_id"], objective.id
        )

    def test_live_gap_sources_exclude_revoked_evidence(
        self,
    ) -> None:
        objective = next(
            objective
            for objective in self.database.get_learning_objectives()
            if objective.id == "lo_attention_resource_scaling"
        )
        with self.database.read() as connection:
            source_question_ids = tuple(
                row["question_id"]
                for row in connection.execute(
                    """SELECT DISTINCT direct.question_id
                       FROM release_question_objectives direct
                       JOIN question_sources source
                         ON source.question_id = direct.question_id
                       JOIN release_questions membership
                         ON membership.release_id = direct.release_id
                        AND membership.question_id = direct.question_id
                       WHERE direct.release_id = ?
                         AND direct.objective_id = ?
                         AND source.source_id = ?
                         AND membership.status
                             IN ('approved', 'calibrated')
                       ORDER BY direct.question_id""",
                    (
                        self.database.get_active_release_id(connection),
                        objective.id,
                        "src_goodfellow_dl_2016",
                    ),
                )
            )
        self.assertTrue(source_question_ids)
        for question_id in source_question_ids:
            self.database.revoke_question(
                question_id,
                "Live-gap regression: source is no longer backed by live evidence.",
            )
        session = self.engine.start_session(
            "learner-1", "Transformers", mode="learn", seed=193
        )
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE sessions
                   SET phase = 'verify', focus_concept_id = ?,
                       focus_misconception_id = NULL,
                       focus_objective_id = ?
                   WHERE id = ?""",
                (
                    objective.primary_concept_id,
                    objective.id,
                    session["id"],
                ),
            )

        self.engine._record_corpus_gap(
            session["id"],
            message="Corpus gap: revoked source-boundary regression.",
            now=datetime.now(timezone.utc),
        )

        with self.database.read() as connection:
            row = connection.execute(
                "SELECT blueprint_json FROM generation_jobs"
            ).fetchone()
        blueprint = json.loads(row["blueprint_json"])
        self.assertEqual(blueprint["learning_objective_id"], objective.id)
        self.assertIn("src_vaswani_attention_2017", blueprint["source_ids"])
        self.assertNotIn(
            "src_goodfellow_dl_2016",
            blueprint["source_ids"],
        )

    def test_live_gap_sources_fall_back_to_live_primary_concept(self) -> None:
        objective = next(
            objective
            for objective in self.database.get_learning_objectives()
            if objective.id == "lo_attention_resource_scaling"
        )
        with self.database.read() as connection:
            direct_question_ids = tuple(
                row["question_id"]
                for row in connection.execute(
                    """SELECT direct.question_id
                       FROM release_question_objectives direct
                       JOIN release_questions membership
                         ON membership.release_id = direct.release_id
                        AND membership.question_id = direct.question_id
                       WHERE direct.release_id = ?
                         AND direct.objective_id = ?
                         AND membership.status IN ('approved', 'calibrated')
                       ORDER BY direct.question_id""",
                    (
                        self.database.get_active_release_id(connection),
                        objective.id,
                    ),
                )
            )
        self.assertTrue(direct_question_ids)
        for question_id in direct_question_ids:
            self.database.revoke_question(
                question_id,
                "Live-gap regression: exhaust direct objective evidence.",
            )

        with self.database.read() as connection:
            expected_fallback_sources = [
                row["source_id"]
                for row in connection.execute(
                    """SELECT DISTINCT source.source_id
                       FROM question_sources source
                       JOIN question_concepts mapping
                         ON mapping.question_id = source.question_id
                       JOIN release_questions membership
                         ON membership.question_id = source.question_id
                       WHERE membership.release_id = ?
                         AND mapping.concept_id = ?
                         AND mapping.role = 'primary'
                         AND membership.status
                             IN ('approved', 'calibrated')
                         AND NOT EXISTS (
                             SELECT 1
                             FROM question_revocations revoked
                             WHERE revoked.question_id = source.question_id
                         )
                       ORDER BY source.source_id LIMIT 8""",
                    (
                        self.database.get_active_release_id(connection),
                        objective.primary_concept_id,
                    ),
                )
            ]
        self.assertTrue(expected_fallback_sources)
        self.assertIn(
            "src_vaswani_attention_2017",
            expected_fallback_sources,
        )

        session = self.engine.start_session(
            "learner-1", "Transformers", mode="learn", seed=197
        )
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE sessions
                   SET phase = 'verify', focus_concept_id = ?,
                       focus_misconception_id = NULL,
                       focus_objective_id = ?
                   WHERE id = ?""",
                (
                    objective.primary_concept_id,
                    objective.id,
                    session["id"],
                ),
            )

        self.engine._record_corpus_gap(
            session["id"],
            message="Corpus gap: primary-concept source fallback regression.",
            now=datetime.now(timezone.utc),
        )

        with self.database.read() as connection:
            row = connection.execute(
                "SELECT blueprint_json FROM generation_jobs"
            ).fetchone()
        blueprint = json.loads(row["blueprint_json"])
        self.assertEqual(blueprint["learning_objective_id"], objective.id)
        self.assertEqual(
            blueprint["source_ids"],
            expected_fallback_sources,
        )

    def test_session_end_is_durable_idempotent_and_blocks_pending_work(self) -> None:
        session = self.start(seed=13)
        presentation = self.engine.next_question(session["id"])
        ended = self.engine.end_session(
            session["id"],
            completed=False,
            reason="test_stop",
            idempotency_key="end-session-once",
        )
        replay = self.engine.end_session(
            session["id"],
            completed=False,
            reason="test_stop",
            idempotency_key="end-session-once",
        )
        self.assertEqual(ended["status"], "abandoned")
        self.assertEqual(replay["status"], "abandoned")
        with self.assertRaises(ExhaustedError):
            self.engine.next_question(session["id"])
        with self.assertRaises(ConflictError):
            self.engine.submit_answer(
                presentation.decision_id,
                presentation.question.correct_option.id,
                idempotency_key="answer-after-end",
            )
        with self.database.read() as connection:
            events = connection.execute(
                "SELECT COUNT(*) AS n FROM events WHERE event_type = 'SessionEnded'"
            ).fetchone()["n"]
        self.assertEqual(events, 1)

    def test_selected_primary_concept_stays_in_scope_unless_exploration_is_labeled(self) -> None:
        session = self.start()
        scope = self.database.topic_scope("t_machine_learning")
        for index in range(6):
            presentation = self.engine.next_question(session["id"])
            if presentation.pedagogical_role == "exploration_probe":
                self.assertNotIn(presentation.question.primary_concept_id, scope)
                self.assertIn(
                    "deliberate_related_topic_probe", presentation.rationale
                )
            else:
                self.assertIn(presentation.question.primary_concept_id, scope)
            self.engine.submit_answer(
                presentation.decision_id,
                presentation.question.correct_option.id,
                idempotency_key=f"scope-{index}",
            )

    def test_cold_learner_can_repair_safely_at_every_graph_root(self) -> None:
        graph = self.database.get_graph()
        for index, root_id in enumerate(sorted(graph.concepts)):
            learner_id = f"all-roots-{index}"
            self.engine.create_learner(learner_id, root_id)
            session = self.engine.start_session(
                learner_id,
                root_id,
                mode="learn",
                seed=700 + index,
                now=self.test_clock,
            )

            presentation = self.next_at(session["id"])

            self.assertIn(
                presentation.question.primary_concept_id,
                graph.learning_scope(root_id),
                root_id,
            )
            wrong = next(
                option for option in presentation.question.options if not option.correct
            )
            self.submit_at(
                presentation.decision_id,
                wrong.id,
                confidence=0.9,
                response_ms=900,
                idempotency_key=f"all-roots-wrong-{index}",
            )

            repair = self.next_at(session["id"])
            self.assertNotEqual(
                repair.question.family_id, presentation.question.family_id, root_id
            )
            self.submit_at(
                repair.decision_id,
                repair.question.correct_option.id,
                confidence=0.9,
                response_ms=900,
                idempotency_key=f"all-roots-repair-{index}",
            )

            verification = self.next_at(session["id"])
            self.assertNotIn(
                verification.question.family_id,
                {presentation.question.family_id, repair.question.family_id},
                root_id,
            )
            self.submit_at(
                verification.decision_id,
                verification.question.correct_option.id,
                confidence=0.9,
                response_ms=900,
                idempotency_key=f"all-roots-verify-{index}",
            )

    def test_wrong_answer_enters_targeted_remediation_without_repeating_item(self) -> None:
        session = self.start(mode="diagnose")
        first = self.next_at(session["id"])
        wrong = next(option for option in first.question.options if not option.correct)
        result = self.submit_at(
            first.decision_id,
            wrong.id,
            confidence=0.8,
            response_ms=1500,
            idempotency_key="answer-1",
        )
        self.assertFalse(result.correct)
        self.assertEqual(result.next_phase, SessionPhase.REMEDIATE)
        self.assertEqual(result.focus_misconception_id, wrong.misconception_id)

        next_presentation = self.next_at(session["id"])
        self.assertNotEqual(next_presentation.question.id, first.question.id)
        self.assertNotEqual(next_presentation.question.family_id, first.question.family_id)

    def test_remediation_success_requires_independent_verification(self) -> None:
        session = self.start()
        first = self.next_at(session["id"])
        wrong = next(option for option in first.question.options if not option.correct)
        self.submit_at(
            first.decision_id,
            wrong.id,
            confidence=0.9,
            response_ms=900,
            idempotency_key="step-1",
        )

        repair = self.next_at(session["id"])
        result = self.submit_at(
            repair.decision_id,
            repair.question.correct_option.id,
            confidence=0.9,
            response_ms=900,
            idempotency_key="step-2",
        )
        self.assertTrue(result.correct)
        self.assertEqual(result.next_phase, SessionPhase.VERIFY)

        verification = self.next_at(session["id"])
        self.assertNotIn(
            verification.question.family_id,
            {first.question.family_id, repair.question.family_id},
        )
        verified = self.submit_at(
            verification.decision_id,
            verification.question.correct_option.id,
            confidence=0.9,
            response_ms=900,
            idempotency_key="step-3",
        )
        self.assertEqual(verified.next_phase, SessionPhase.LEARN)
        self.assertIsNone(verified.focus_misconception_id)

    def test_prerequisite_repair_returns_to_original_unresolved_goal(self) -> None:
        session = self.engine.start_session(
            "learner-1",
            "c_clustering",
            mode="learn",
            seed=13,
            now=self.test_clock,
        )
        trigger = self.next_at(session["id"])
        trigger_wrong = next(
            option for option in trigger.question.options if not option.correct
        )
        first_failure = self.submit_at(
            trigger.decision_id,
            trigger_wrong.id,
            confidence=0.9,
            response_ms=900,
            idempotency_key="parent-trigger",
        )
        original_concept = first_failure.focus_concept_id
        original_misconception = first_failure.focus_misconception_id

        repair = self.next_at(session["id"])
        repair_wrong = next(
            option for option in repair.question.options if not option.correct
        )
        descended = self.submit_at(
            repair.decision_id,
            repair_wrong.id,
            confidence=0.9,
            response_ms=900,
            idempotency_key="parent-descend",
        )
        descended_session = self.database.get_session(session["id"])
        self.assertEqual(descended.focus_concept_id, "c_feature_scaling")
        self.assertEqual(
            descended.transition_reason, "descend_to_evidence_boundary"
        )
        self.assertIsNotNone(descended.boundary_decision)
        assert descended.boundary_decision is not None
        self.assertEqual(
            descended.boundary_decision["selected_concept_id"],
            "c_feature_scaling",
        )
        self.assertEqual(
            descended.boundary_decision["selected"]["concept_id"],
            "c_feature_scaling",
        )
        self.assertTrue(descended.boundary_decision["candidates"])
        self.assertEqual(
            descended_session["remediation_path"],
            [
                {
                    "concept_id": original_concept,
                    "misconception_id": original_misconception,
                }
            ],
        )

        prerequisite_probe = self.next_at(session["id"])
        self.submit_at(
            prerequisite_probe.decision_id,
            prerequisite_probe.question.correct_option.id,
            confidence=0.9,
            response_ms=900,
            idempotency_key="parent-prerequisite-repair",
        )
        prerequisite_verify = self.next_at(session["id"])
        returned = self.submit_at(
            prerequisite_verify.decision_id,
            prerequisite_verify.question.correct_option.id,
            confidence=0.9,
            response_ms=900,
            idempotency_key="parent-prerequisite-verify",
        )
        returned_session = self.database.get_session(session["id"])
        self.assertEqual(returned.next_phase, SessionPhase.VERIFY)
        self.assertEqual(returned.focus_concept_id, original_concept)
        self.assertEqual(returned.focus_misconception_id, original_misconception)
        self.assertEqual(returned_session["remediation_path"], [])
        self.assertEqual(returned_session["remediation_depth"], 1)
        self.assertEqual(
            returned.transition_reason, "prerequisite_verified_resume_parent"
        )

        parent_recheck = self.next_at(session["id"])
        self.assertEqual(parent_recheck.pedagogical_role, "verification")
        self.assertEqual(
            parent_recheck.question.primary_concept_id, original_concept
        )

    def test_uncertain_or_instant_success_requires_independent_confirmation(self) -> None:
        scenarios = (
            ("low-confidence", 0.10, 900),
            ("instant", 0.90, 0),
        )
        for index, (label, confidence, response_ms) in enumerate(scenarios):
            learner_id = f"uncertain-{index}"
            self.engine.create_learner(learner_id, label)
            session = self.engine.start_session(
                learner_id,
                "t_machine_learning",
                mode="learn",
                seed=127 + index,
                now=self.test_clock,
            )
            first = self.next_at(session["id"])
            uncertain = self.submit_at(
                first.decision_id,
                first.question.correct_option.id,
                confidence=confidence,
                response_ms=response_ms,
                idempotency_key=f"uncertain-first-{index}",
            )
            self.assertEqual(uncertain.next_phase, SessionPhase.VERIFY, label)
            self.assertEqual(
                uncertain.focus_concept_id,
                first.question.primary_concept_id,
                label,
            )
            confirmation = self.next_at(session["id"])
            self.assertNotEqual(
                confirmation.question.family_id, first.question.family_id, label
            )
            confirmed = self.submit_at(
                confirmation.decision_id,
                confirmation.question.correct_option.id,
                confidence=0.90,
                response_ms=900,
                idempotency_key=f"uncertain-confirm-{index}",
            )
            self.assertEqual(confirmed.next_phase, SessionPhase.LEARN, label)

    def test_impossible_response_time_is_atomic_and_retryable(self) -> None:
        learner_id = "authoritative-response-window"
        self.engine.create_learner(learner_id)
        session = self.engine.start_session(
            learner_id,
            "t_machine_learning",
            seed=811,
            now=START,
        )
        presentation = self.engine.next_question(session["id"], now=START)
        with self.database.read() as connection:
            before = {
                "events": connection.execute(
                    "SELECT COUNT(*) AS n FROM events"
                ).fetchone()["n"],
                "attempts": connection.execute(
                    "SELECT COUNT(*) AS n FROM attempts"
                ).fetchone()["n"],
                "learner_revision": connection.execute(
                    "SELECT revision FROM learners WHERE id = ?",
                    (learner_id,),
                ).fetchone()["revision"],
                "session": dict(
                    connection.execute(
                        "SELECT * FROM sessions WHERE id = ?",
                        (session["id"],),
                    ).fetchone()
                ),
            }

        key = "authoritative-response-window"
        with self.assertRaisesRegex(
            ValidationError, "cannot exceed the authoritative time"
        ):
            self.engine.submit_answer(
                presentation.decision_id,
                presentation.question.correct_option.id,
                confidence=0.9,
                response_ms=1000,
                idempotency_key=key,
                now=START + timedelta(milliseconds=999),
            )

        with self.database.read() as connection:
            after = {
                "events": connection.execute(
                    "SELECT COUNT(*) AS n FROM events"
                ).fetchone()["n"],
                "attempts": connection.execute(
                    "SELECT COUNT(*) AS n FROM attempts"
                ).fetchone()["n"],
                "learner_revision": connection.execute(
                    "SELECT revision FROM learners WHERE id = ?",
                    (learner_id,),
                ).fetchone()["revision"],
                "session": dict(
                    connection.execute(
                        "SELECT * FROM sessions WHERE id = ?",
                        (session["id"],),
                    ).fetchone()
                ),
                "consumed_at": connection.execute(
                    "SELECT consumed_at FROM decisions WHERE id = ?",
                    (presentation.decision_id,),
                ).fetchone()["consumed_at"],
                "idempotency_events": connection.execute(
                    "SELECT COUNT(*) AS n FROM events WHERE idempotency_key = ?",
                    (key,),
                ).fetchone()["n"],
            }
        self.assertEqual(after["events"], before["events"])
        self.assertEqual(after["attempts"], before["attempts"])
        self.assertEqual(
            after["learner_revision"], before["learner_revision"]
        )
        self.assertEqual(after["session"], before["session"])
        self.assertIsNone(after["consumed_at"])
        self.assertEqual(after["idempotency_events"], 0)

        accepted = self.engine.submit_answer(
            presentation.decision_id,
            presentation.question.correct_option.id,
            confidence=0.9,
            response_ms=1000,
            idempotency_key=key,
            now=START + timedelta(seconds=1),
        )
        # A retry returns the already validated immutable result; its arrival
        # clock cannot retroactively make the original response impossible.
        replayed = self.engine.submit_answer(
            presentation.decision_id,
            presentation.question.correct_option.id,
            confidence=0.9,
            response_ms=1000,
            idempotency_key=key,
            now=START,
        )
        self.assertEqual(replayed.interaction_id, accepted.interaction_id)
        self.assertTrue(replayed.idempotent_replay)

    def test_missing_and_subthreshold_confidence_do_not_localize_error(self) -> None:
        for index, confidence in enumerate((None, 0.49)):
            with self.subTest(confidence=confidence):
                learner_id = f"uncertain-error-{index}"
                self.engine.create_learner(learner_id)
                session = self.engine.start_session(
                    learner_id,
                    "t_transformers",
                    seed=2,
                    now=START,
                )
                presentation = self.engine.next_question(
                    session["id"], now=START
                )
                wrong = next(
                    option
                    for option in presentation.question.options
                    if not option.correct
                )
                result = self.engine.submit_answer(
                    presentation.decision_id,
                    wrong.id,
                    confidence=confidence,
                    response_ms=1000,
                    idempotency_key=f"uncertain-error-{index}",
                    now=START + timedelta(seconds=1),
                )

                self.assertEqual(result.next_phase, SessionPhase.VERIFY)
                self.assertEqual(
                    result.transition_reason,
                    "uncertain_main_requires_independent_diagnostic",
                )
                self.assertEqual(
                    result.focus_objective_id,
                    presentation.question.objective_id,
                )
                self.assertIsNone(result.focus_misconception_id)
                self.assertEqual(
                    self.database.get_session(session["id"])[
                        "remediation_path"
                    ],
                    [],
                )

    def test_only_named_error_can_cross_objective_diagnose(self) -> None:
        scenarios = (
            ("named", 0.80, True),
            ("generic", 0.79, False),
        )
        for index, (label, confidence, named_error) in enumerate(scenarios):
            with self.subTest(label=label):
                learner_id = f"{label}-diagnostic-error"
                self.engine.create_learner(learner_id)
                session = self.engine.start_session(
                    learner_id,
                    "t_transformers",
                    seed=2,
                    now=START,
                )
                presentation = self.engine.next_question(
                    session["id"], now=START
                )
                wrong = next(
                    option
                    for option in presentation.question.options
                    if not option.correct
                    and option.diagnostic_objective_id
                    != presentation.question.objective_id
                )
                result = self.engine.submit_answer(
                    presentation.decision_id,
                    wrong.id,
                    confidence=confidence,
                    response_ms=1000,
                    idempotency_key=f"{label}-diagnostic-error",
                    now=START + timedelta(seconds=1),
                )

                self.assertEqual(result.next_phase, SessionPhase.REMEDIATE)
                expected_objective_id = (
                    wrong.diagnostic_objective_id
                    if named_error
                    else presentation.question.objective_id
                )
                expected_misconception_id = (
                    wrong.misconception_id if named_error else None
                )
                expected_reason = (
                    "cross_objective_diagnostic_focus"
                    if named_error
                    else "credible_generic_error_focus"
                )
                expected_path_depth = 1 if named_error else 0
                self.assertEqual(
                    result.focus_objective_id,
                    expected_objective_id,
                )
                self.assertEqual(
                    result.focus_misconception_id,
                    expected_misconception_id,
                )
                self.assertEqual(
                    result.transition_reason,
                    expected_reason,
                )
                self.assertEqual(
                    len(
                        self.database.get_session(session["id"])[
                            "remediation_path"
                        ]
                    ),
                    expected_path_depth,
                )

    def test_repeated_uncertainty_escalates_once_then_exits_boundedly(self) -> None:
        learner_id = "bounded-uncertainty"
        self.engine.create_learner(learner_id)
        session = self.engine.start_session(
            learner_id,
            "t_transformers",
            seed=2,
            now=START,
        )
        main = self.engine.next_question(session["id"], now=START)
        first = self.engine.submit_answer(
            main.decision_id,
            None,
            confidence=0.2,
            response_ms=1000,
            idempotency_key="bounded-uncertainty-main",
            now=START + timedelta(minutes=1),
        )
        self.assertEqual(first.next_phase, SessionPhase.VERIFY)
        self.assertIsNone(first.boundary_decision)

        diagnostic = self.engine.next_question(
            session["id"], now=START + timedelta(minutes=2)
        )
        repeated = self.engine.submit_answer(
            diagnostic.decision_id,
            None,
            confidence=0.2,
            response_ms=1000,
            idempotency_key="bounded-uncertainty-diagnostic",
            now=START + timedelta(minutes=3),
        )
        self.assertEqual(repeated.next_phase, SessionPhase.REMEDIATE)
        self.assertEqual(
            repeated.transition_reason,
            "repeated_uncertainty_requires_bounded_remediation",
        )
        self.assertIsNone(repeated.focus_misconception_id)
        self.assertIsNone(repeated.boundary_decision)

        remediation = self.engine.next_question(
            session["id"], now=START + timedelta(minutes=4)
        )
        bounded = self.engine.submit_answer(
            remediation.decision_id,
            None,
            confidence=0.2,
            response_ms=1000,
            idempotency_key="bounded-uncertainty-remediation",
            now=START + timedelta(minutes=5),
        )
        self.assertEqual(bounded.next_phase, SessionPhase.LEARN)
        self.assertEqual(
            bounded.transition_reason, "uncertain_remediation_bounded_exit"
        )
        self.assertIsNone(bounded.boundary_decision)
        current = self.database.get_session(session["id"])
        self.assertEqual(current["remediation_depth"], 0)
        self.assertEqual(current["remediation_path"], [])

    def test_uncertain_attempt_does_not_dilute_boundary_failure_rate(self) -> None:
        learner_id = "boundary-response-pressure"
        self.engine.create_learner(learner_id)
        session = self.engine.start_session(
            learner_id,
            "t_transformers",
            seed=2,
            now=START,
        )
        main = self.engine.next_question(session["id"], now=START)
        uncertain = self.engine.submit_answer(
            main.decision_id,
            None,
            confidence=0.2,
            response_ms=1000,
            idempotency_key="boundary-uncertain",
            now=START + timedelta(minutes=1),
        )
        self.assertEqual(uncertain.next_phase, SessionPhase.VERIFY)

        verification = self.engine.next_question(
            session["id"], now=START + timedelta(minutes=2)
        )
        wrong = next(
            option
            for option in verification.question.options
            if not option.correct
            and option.diagnostic_objective_id
            == "lo_causal_visibility"
        )
        credible_failure = self.engine.submit_answer(
            verification.decision_id,
            wrong.id,
            confidence=0.60,
            response_ms=1000,
            idempotency_key="boundary-credible-error",
            now=START + timedelta(minutes=3),
        )
        self.assertNotEqual(
            credible_failure.transition_reason,
            "uncertain_main_requires_independent_diagnostic",
        )

        with self.database.read() as connection:
            boundary = self.engine._declared_objective_boundary(
                connection,
                session_id=session["id"],
                learner_id=learner_id,
                release_id=session["corpus_release_id"],
                focus_objective_id="lo_incremental_kv_cache",
                now=START + timedelta(minutes=4),
            )
        self.assertIsNotNone(boundary)
        self.assertIsNotNone(boundary["boundary_decision"])
        failed_objective = next(
            candidate
            for candidate in boundary["boundary_decision"]["candidates"]
            if candidate["objective_id"]
            == verification.question.objective_id
        )
        # One low-confidence abstention plus one credible error must exert the
        # pressure of exactly one error, not a diluted 1/2 raw-attempt rate.
        self.assertEqual(failed_objective["recent_failure_rate"], 1.0)

    def test_exploration_gate_rejects_uncertain_successes(self) -> None:
        session = {
            "topic_id": "t_transformers",
            "exploration_mode": "adaptive",
            "focus_concept_id": None,
            "focus_misconception_id": None,
            "focus_objective_id": None,
            "step": 3,
        }

        def recent(
            *, confidence: float | None, response_ms: int | None
        ) -> list[dict]:
            return [
                {
                    "correct": True,
                    "pedagogical_role": "main",
                    "hint_count": 0,
                    "confidence": confidence,
                    "response_ms": response_ms,
                    "selected_option_id": "option-a",
                    "learner_model_version": (
                        self.engine.learner_model.model_version
                    ),
                    "question_id": f"question-{index}",
                }
                for index in range(3)
            ]

        self.assertTrue(
            self.engine.policy._should_explore(
                session,
                SessionPhase.LEARN,
                recent(confidence=0.90, response_ms=900),
            )
        )
        for confidence, response_ms in (
            (None, 900),
            (0.49, 900),
            (0.90, None),
        ):
            with self.subTest(
                confidence=confidence, response_ms=response_ms
            ):
                self.assertFalse(
                    self.engine.policy._should_explore(
                        session,
                        SessionPhase.LEARN,
                        recent(
                            confidence=confidence,
                            response_ms=response_ms,
                        ),
                    )
                )

    def test_repeated_instant_success_exits_inconclusive_verification_boundedly(self) -> None:
        self.engine.create_learner("instant-repeat", "Repeated instant responses")
        session = self.engine.start_session(
            "instant-repeat", "c_feature_scaling", mode="learn", seed=313
        )
        first = self.engine.next_question(session["id"])
        routed = self.engine.submit_answer(
            first.decision_id,
            first.question.correct_option.id,
            confidence=0.95,
            response_ms=0,
            idempotency_key="instant-repeat-main",
        )
        self.assertEqual(routed.next_phase, SessionPhase.VERIFY)

        verification = self.engine.next_question(session["id"])
        self.assertNotEqual(verification.question.family_id, first.question.family_id)
        inconclusive = self.engine.submit_answer(
            verification.decision_id,
            verification.question.correct_option.id,
            confidence=0.95,
            response_ms=0,
            idempotency_key="instant-repeat-verification",
        )

        self.assertEqual(inconclusive.next_phase, SessionPhase.LEARN)
        current = self.database.get_session(session["id"])
        self.assertEqual(current["remediation_depth"], 0)
        self.assertIsNone(current["focus_concept_id"])
        with self.database.read() as connection:
            certified = connection.execute(
                """SELECT COUNT(*) AS n FROM learner_skill_families
                   WHERE learner_id='instant-repeat'"""
            ).fetchone()["n"]
        self.assertEqual(certified, 0)

    def test_review_verify_failure_without_deeper_prerequisite_exits_bounded_tunnel(self) -> None:
        session = self.engine.start_session(
            "learner-1",
            "c_probability_reasoning",
            mode="review",
            seed=17,
            now=self.test_clock,
        )
        first = self.next_at(session["id"])
        wrong = next(option for option in first.question.options if not option.correct)
        self.submit_at(
            first.decision_id,
            wrong.id,
            confidence=0.9,
            response_ms=900,
            idempotency_key="review-1",
        )

        repair = self.next_at(session["id"])
        self.submit_at(
            repair.decision_id,
            repair.question.correct_option.id,
            confidence=0.9,
            response_ms=900,
            idempotency_key="review-2",
        )
        self.assertEqual(self.database.get_session(session["id"])["phase"], "verify")

        verify = self.next_at(session["id"])
        verify_wrong = next(option for option in verify.question.options if not option.correct)
        failed = self.submit_at(
            verify.decision_id,
            verify_wrong.id,
            confidence=0.9,
            response_ms=900,
            idempotency_key="review-3",
        )
        self.assertEqual(failed.next_phase, SessionPhase.REVIEW)
        self.assertEqual(self.database.get_session(session["id"])["remediation_depth"], 0)
        self.assertIsNone(failed.focus_concept_id)

    def test_answer_submission_is_idempotent(self) -> None:
        session = self.start()
        presentation = self.engine.next_question(session["id"])
        key = "stable-answer-key"
        first = self.engine.submit_answer(
            presentation.decision_id,
            presentation.question.correct_option.id,
            idempotency_key=key,
        )
        second = self.engine.submit_answer(
            presentation.decision_id,
            presentation.question.correct_option.id,
            idempotency_key=key,
        )
        self.assertEqual(first.interaction_id, second.interaction_id)
        self.assertTrue(second.idempotent_replay)
        with self.database.read() as connection:
            attempts = connection.execute("SELECT COUNT(*) AS n FROM attempts").fetchone()["n"]
        self.assertEqual(attempts, 1)

    def test_integer_confidence_endpoints_are_canonical_floats(self) -> None:
        for confidence in (0, 1):
            with self.subTest(confidence=confidence):
                learner_id = f"integer-confidence-{confidence}"
                self.engine.create_learner(learner_id)
                session = self.engine.start_session(
                    learner_id,
                    "t_machine_learning",
                    mode="learn",
                    seed=211 + confidence,
                    now=self.test_clock,
                )
                presentation = self.next_at(session["id"])
                key = f"integer-confidence-answer-{confidence}"
                first = self.submit_at(
                    presentation.decision_id,
                    presentation.question.correct_option.id,
                    confidence=confidence,
                    response_ms=800,
                    idempotency_key=key,
                )
                replay = self.engine.submit_answer(
                    presentation.decision_id,
                    presentation.question.correct_option.id,
                    confidence=float(confidence),
                    response_ms=800,
                    idempotency_key=key,
                )
                self.assertTrue(replay.idempotent_replay)
                self.assertEqual(replay.interaction_id, first.interaction_id)
                with self.database.read() as connection:
                    event_confidence = json.loads(
                        connection.execute(
                            "SELECT payload_json FROM events WHERE idempotency_key = ?",
                            (key,),
                        ).fetchone()["payload_json"]
                    )["confidence"]
                    attempt_confidence = connection.execute(
                        "SELECT confidence FROM attempts WHERE decision_id = ?",
                        (presentation.decision_id,),
                    ).fetchone()["confidence"]
                self.assertIsInstance(event_confidence, float)
                self.assertEqual(event_confidence, float(confidence))
                self.assertIsInstance(attempt_confidence, float)
                self.assertEqual(attempt_confidence, float(confidence))

        report = self.database.verify_integrity()
        self.assertTrue(report["ok"], report["errors"])

    def test_idempotency_covers_the_full_effectful_answer_payload_and_outcome(self) -> None:
        session = self.start(seed=31)
        presentation = self.next_at(session["id"])
        option_id = presentation.question.correct_option.id
        key = "full-answer-command"
        inputs = {
            "confidence": 0.22,
            "response_ms": 321,
            "hint_count": 1,
            "feedback_shown": False,
        }
        first = self.submit_at(
            presentation.decision_id,
            option_id,
            idempotency_key=key,
            **inputs,
        )
        replay = self.engine.submit_answer(
            presentation.decision_id,
            option_id,
            idempotency_key=key,
            **inputs,
        )
        self.assertTrue(replay.idempotent_replay)
        self.assertEqual(replay.interaction_id, first.interaction_id)
        self.assertEqual(replay.correct, first.correct)
        self.assertEqual(replay.selected_option.id, first.selected_option.id)
        self.assertEqual(replay.correct_option.id, first.correct_option.id)
        self.assertEqual(replay.next_phase, first.next_phase)
        self.assertEqual(replay.focus_concept_id, first.focus_concept_id)
        self.assertEqual(replay.focus_misconception_id, first.focus_misconception_id)
        self.assertEqual(replay.state_changes, first.state_changes)
        self.assertEqual(replay.transition_reason, first.transition_reason)
        self.assertEqual(replay.boundary_decision, first.boundary_decision)

        changed_commands = (
            {**inputs, "confidence": 0.91},
            {**inputs, "response_ms": 999},
            {**inputs, "hint_count": 2},
            {**inputs, "feedback_shown": True},
        )
        for changed in changed_commands:
            with self.subTest(changed=changed), self.assertRaises(ConflictError):
                self.engine.submit_answer(
                    presentation.decision_id,
                    option_id,
                    idempotency_key=key,
                    **changed,
                )

        with self.database.read() as connection:
            attempt = connection.execute(
                "SELECT command_hash, outcome_json FROM attempts WHERE decision_id = ?",
                (presentation.decision_id,),
            ).fetchone()
            event_payload = json.loads(
                connection.execute(
                    "SELECT payload_json FROM events WHERE idempotency_key = ?", (key,)
                ).fetchone()["payload_json"]
            )
        self.assertRegex(attempt["command_hash"], r"^[0-9a-f]{64}$")
        outcome = json.loads(attempt["outcome_json"])
        self.assertEqual(outcome["next_phase"], first.next_phase.value)
        self.assertEqual(outcome["state_changes"], list(first.state_changes))
        for field, value in inputs.items():
            self.assertEqual(event_payload[field], value)

    def test_distractor_updates_misconception_as_hypothesis(self) -> None:
        session = self.start()
        presentation = self.next_at(session["id"])
        wrong = next(option for option in presentation.question.options if not option.correct)
        self.submit_at(
            presentation.decision_id,
            wrong.id,
            confidence=0.90,
            response_ms=1_000,
            idempotency_key="mis-1",
        )
        belief = self.database.get_misconception_beliefs("learner-1")[wrong.misconception_id]
        self.assertGreater(belief.probability, 0.10)
        self.assertLess(belief.probability, 0.90)

    def test_elapsed_time_reduces_retrievability_and_widens_uncertainty(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        session = self.start()
        presentation = self.engine.next_question(session["id"], now=start)
        result = self.engine.submit_answer(
            presentation.decision_id,
            presentation.question.correct_option.id,
            idempotency_key="time-1",
            now=start,
        )
        concept_id = result.state_changes[0]["concept_id"]
        stored = self.database.get_skill_states("learner-1")[concept_id]
        concept = self.database.get_graph().concepts[concept_id]
        projected = self.engine.learner_model.project_state(
            stored, concept, start + timedelta(days=180)
        )
        self.assertLess(projected.mastery_probability, stored.mastery_probability)
        self.assertGreater(projected.variance, stored.variance)

    def test_event_stream_is_append_only_and_hash_chained(self) -> None:
        session = self.start()
        presentation = self.engine.next_question(session["id"])
        self.engine.submit_answer(
            presentation.decision_id,
            presentation.question.correct_option.id,
            idempotency_key="chain-1",
        )
        with self.database.read() as connection:
            events = connection.execute(
                "SELECT * FROM events WHERE stream_id = ? ORDER BY stream_version",
                ("learner:learner-1",),
            ).fetchall()
        self.assertEqual([row["stream_version"] for row in events], list(range(1, len(events) + 1)))
        for previous, current in zip(events, events[1:], strict=False):
            self.assertEqual(current["previous_hash"], previous["payload_hash"])
        response = next(row for row in events if row["event_type"] == "ResponseSubmitted")
        projection = next(row for row in events if row["event_type"] == "LearnerProjectionAdvanced")
        self.assertEqual(response["schema_version"], 2)
        self.assertEqual(projection["schema_version"], 2)
        self.assertEqual(json.loads(projection["payload_json"])["response_event_id"], response["event_id"])
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_event_and_attempt_rows_are_immutable(self) -> None:
        session = self.start()
        presentation = self.engine.next_question(session["id"])
        self.engine.submit_answer(
            presentation.decision_id,
            presentation.question.correct_option.id,
            idempotency_key="tamper-1",
        )
        forbidden_mutations = (
            "UPDATE events SET occurred_at = '1900-01-01T00:00:00+00:00'",
            "DELETE FROM events WHERE event_type = 'LearnerProjectionAdvanced'",
            "UPDATE attempts SET confidence = 0.99",
            "DELETE FROM attempts",
        )
        for statement in forbidden_mutations:
            with self.subTest(statement=statement), self.assertRaises(sqlite3.IntegrityError):
                with self.database.transaction() as connection:
                    connection.execute(statement)
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_integrity_verifier_detects_unhashed_envelope_tampering(self) -> None:
        session = self.start()
        presentation = self.engine.next_question(session["id"])
        self.engine.submit_answer(
            presentation.decision_id,
            presentation.question.correct_option.id,
            idempotency_key="envelope-tamper",
        )
        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER events_no_update")
            connection.execute(
                """UPDATE events SET occurred_at = '1900-01-01T00:00:00+00:00'
                   WHERE idempotency_key = 'envelope-tamper'"""
            )
        report = self.database.verify_integrity()
        self.assertFalse(report["ok"])
        self.assertTrue(any("hash mismatch" in error for error in report["errors"]))

    def test_integrity_verifier_detects_learner_projection_tampering(self) -> None:
        session = self.start(seed=109)
        presentation = self.next_at(session["id"])
        self.submit_at(
            presentation.decision_id,
            presentation.question.correct_option.id,
            confidence=0.8,
            response_ms=900,
            idempotency_key="projection-tamper",
        )
        self.assertTrue(self.database.verify_integrity()["ok"])
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE skill_states SET mean = mean + 3.0,
                          evidence_mass = evidence_mass + 99.0
                   WHERE learner_id = 'learner-1'"""
            )
        report = self.database.verify_integrity()
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("projection hash mismatch" in error for error in report["errors"]),
            report["errors"],
        )

    def test_integrity_verifier_detects_projection_before_first_projection_event(self) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO skill_states(
                       learner_id, concept_id, mean, variance, stability_hours,
                       exposures, last_seen_at, next_review_at, evidence_mass,
                       as_of_event_id, model_version
                   ) VALUES ('learner-1', 'c_probability_reasoning', 2.5, 1.0,
                             24.0, 1, NULL, NULL, 1.0, NULL, 'forged')"""
            )

        report = self.database.verify_integrity()
        self.assertFalse(report["ok"])
        self.assertTrue(
            any(
                "mutable projection rows exist without a LearnerProjectionAdvanced event"
                in error
                for error in report["errors"]
            ),
            report["errors"],
        )

    def test_integrity_verifier_detects_stream_tail_truncation(self) -> None:
        session = self.start()
        presentation = self.engine.next_question(session["id"])
        self.engine.submit_answer(
            presentation.decision_id,
            presentation.question.correct_option.id,
            idempotency_key="tail-truncation",
        )
        stream_id = "learner:learner-1"
        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER events_no_delete")
            tail = connection.execute(
                """SELECT event_id FROM events WHERE stream_id = ?
                   ORDER BY stream_version DESC LIMIT 1""",
                (stream_id,),
            ).fetchone()
            connection.execute("DELETE FROM events WHERE event_id = ?", (tail["event_id"],))
        report = self.database.verify_integrity(stream_id)
        self.assertFalse(report["ok"])
        self.assertTrue(any("head" in error.lower() for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
