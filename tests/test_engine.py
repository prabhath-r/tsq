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

from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.errors import ConflictError, ExhaustedError
from tsq.models import SessionPhase
from tsq.store import SCHEMA_VERSION, Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"


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

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def start(self, *, mode: str = "learn", seed: int = 17):
        return self.engine.start_session(
            "learner-1", "t_machine_learning", mode=mode, seed=seed
        )

    def downgrade_current_database_to_v3(self) -> None:
        """Strip v4 additions while retaining populated v3 learner/event data."""
        with self.database.transaction() as connection:
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
            connection.execute("DROP TABLE learning_actions")
            connection.execute("DROP TABLE learning_artifacts")

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
                "pedagogical_role",
                "focus_valid",
                "invalidated_at",
                "invalidation_reason",
            ):
                connection.execute(f"ALTER TABLE decisions DROP COLUMN {column}")
            connection.execute(
                "ALTER TABLE sessions DROP COLUMN remediation_path_json"
            )
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

            for event in connection.execute(
                """SELECT event_id, payload_json FROM events
                   WHERE event_type = 'LearnerProjectionAdvanced'"""
            ).fetchall():
                payload = json.loads(event["payload_json"])
                payload.pop("projection_hash", None)
                payload.pop("learner_revision", None)
                connection.execute(
                    "UPDATE events SET payload_json = ? WHERE event_id = ?",
                    (
                        json.dumps(payload, sort_keys=True, separators=(",", ":")),
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
        self.assertEqual(SCHEMA_VERSION, 8)
        self.assertEqual(value, str(SCHEMA_VERSION))

    def test_topic_session_preserves_continuity_then_explores_explicitly(self) -> None:
        session = self.engine.start_session(
            "learner-1", "Transformers", mode="learn", seed=9
        )
        self.assertEqual(session["topic_id"], "t_transformers")
        self.assertEqual(session["exploration_mode"], "adaptive")
        owned = self.database.topic_owned_concepts("t_transformers")

        for index in range(3):
            presentation = self.engine.next_question(session["id"])
            decision = self.database.recent_decisions(session["id"], 1)[0]
            self.assertEqual(decision["pedagogical_role"], "main")
            self.assertIn(presentation.question.primary_concept_id, owned)
            self.engine.submit_answer(
                presentation.decision_id,
                presentation.question.correct_option.id,
                confidence=0.9,
                response_ms=1800,
                idempotency_key=f"topic-continuity-{index}",
            )

        exploration = self.engine.next_question(session["id"])
        decision = self.database.recent_decisions(session["id"], 1)[0]
        self.assertEqual(decision["pedagogical_role"], "exploration_probe")
        self.assertNotIn(exploration.question.primary_concept_id, owned)
        self.assertIn("deliberate_related_topic_probe", exploration.rationale)

        wrong = next(option for option in exploration.question.options if not option.correct)
        result = self.engine.submit_answer(
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
        repair = self.engine.next_question(session["id"])
        repair_decision = self.database.recent_decisions(session["id"], 1)[0]
        self.assertEqual(repair_decision["pedagogical_role"], "remediation_probe")
        self.assertNotEqual(repair.question.family_id, exploration.question.family_id)
        self.assertIn(result.focus_misconception_id, repair.question.misconception_ids)

    def test_session_report_exposes_time_difficulty_and_uncertainty_paths(self) -> None:
        session = self.engine.start_session(
            "learner-1", "LLM Agents", mode="learn", seed=21
        )
        first = self.engine.next_question(session["id"])
        self.engine.submit_answer(
            first.decision_id,
            first.question.correct_option.id,
            confidence=0.85,
            response_ms=2400,
            idempotency_key="report-first",
        )
        second = self.engine.next_question(session["id"])
        self.engine.submit_answer(
            second.decision_id,
            None,
            confidence=0.2,
            response_ms=5100,
            idempotency_key="report-second",
        )
        self.engine.end_session(
            session["id"], status="completed", reason="report_test"
        )

        report = self.engine.session_report(session["id"])

        self.assertEqual(report["topic"]["id"], "t_llm_agents")
        self.assertEqual(report["questions_answered"], 2)
        self.assertEqual(report["abstained"], 1)
        self.assertEqual(report["response_time"]["active_seconds"], 7.5)
        self.assertIsNotNone(report["difficulty"]["average"])
        self.assertEqual(report["unique_families"], 2)
        self.assertTrue(report["concept_changes"])
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

    def test_populated_v3_database_is_migrated_to_v4(self) -> None:
        session = self.start(seed=29)
        presentation = self.engine.next_question(session["id"])
        self.engine.submit_answer(
            presentation.decision_id,
            presentation.question.correct_option.id,
            confidence=0.72,
            response_ms=840,
            idempotency_key="migration-v3-answer",
        )
        self.downgrade_current_database_to_v3()

        self.database.initialize()

        with self.database.read() as connection:
            version = connection.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()["value"]
            active_release = connection.execute(
                "SELECT value FROM meta WHERE key = 'active_corpus_release'"
            ).fetchone()["value"]
            self.assertEqual(version, str(SCHEMA_VERSION))
            self.assertTrue(active_release.startswith("rel_"))
            for table in ("concepts", "misconceptions", "sources"):
                missing = connection.execute(
                    f"SELECT COUNT(*) AS n FROM {table} WHERE content_hash IS NULL"
                ).fetchone()["n"]
                self.assertEqual(missing, 0, table)
            migrated_session = connection.execute(
                "SELECT corpus_release_id, revision FROM sessions WHERE id = ?",
                (session["id"],),
            ).fetchone()
            self.assertEqual(migrated_session["corpus_release_id"], active_release)
            self.assertEqual(migrated_session["revision"], 0)
            migrated_decision = connection.execute(
                """SELECT question_version, question_content_hash, question_status,
                          evidence_weight, corpus_release_id, session_revision,
                          learner_revision, pedagogical_role, focus_valid
                   FROM decisions WHERE id = ?""",
                (presentation.decision_id,),
            ).fetchone()
            self.assertTrue(all(value is not None for value in migrated_decision))
            self.assertEqual(migrated_decision["corpus_release_id"], active_release)
            migrated_attempt = connection.execute(
                "SELECT command_hash FROM attempts WHERE decision_id = ?",
                (presentation.decision_id,),
            ).fetchone()
            self.assertRegex(migrated_attempt["command_hash"], r"^[0-9a-f]{64}$")
            trigger_names = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger'"
                )
            }
        self.assertTrue(
            {
                "events_no_update",
                "events_no_delete",
                "attempts_validate_insert",
                "attempts_no_update",
                "attempts_no_delete",
            }.issubset(trigger_names)
        )
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_incomplete_v4_release_history_fails_closed(self) -> None:
        with self.database.transaction() as connection:
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
            self.engine.next_question(first_session["id"]),
            self.engine.next_question(second_session["id"]),
        )

        def submit(index: int):
            presentation = presentations[index]
            try:
                result = self.engine.submit_answer(
                    presentation.decision_id,
                    presentation.question.correct_option.id,
                    confidence=0.9,
                    response_ms=900,
                    idempotency_key=f"concurrent-cross-session-{index}",
                )
                return "applied", result.interaction_id
            except ConflictError as exc:
                return "stale", str(exc)

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(submit, range(2)))
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

    def test_parallel_session_stale_decision_is_rejected_and_invalidated(self) -> None:
        first_session = self.start(seed=103)
        second_session = self.start(seed=107)
        first = self.engine.next_question(first_session["id"])
        stale = self.engine.next_question(second_session["id"])

        self.engine.submit_answer(
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

        replacement = self.engine.next_question(second_session["id"])
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
                "SELECT blueprint_json FROM generation_jobs"
            ).fetchone()
        self.assertIsNotNone(row)
        blueprint = json.loads(row["blueprint_json"])
        self.assertIn(
            blueprint["concept_id"],
            self.database.topic_owned_concepts("t_transformers"),
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
                learner_id, root_id, mode="learn", seed=700 + index
            )

            presentation = self.engine.next_question(session["id"])

            self.assertIn(
                presentation.question.primary_concept_id,
                graph.learning_scope(root_id),
                root_id,
            )
            wrong = next(
                option for option in presentation.question.options if not option.correct
            )
            self.engine.submit_answer(
                presentation.decision_id,
                wrong.id,
                confidence=0.9,
                response_ms=900,
                idempotency_key=f"all-roots-wrong-{index}",
            )

            repair = self.engine.next_question(session["id"])
            self.assertNotEqual(
                repair.question.family_id, presentation.question.family_id, root_id
            )
            self.engine.submit_answer(
                repair.decision_id,
                repair.question.correct_option.id,
                confidence=0.9,
                response_ms=900,
                idempotency_key=f"all-roots-repair-{index}",
            )

            verification = self.engine.next_question(session["id"])
            self.assertNotIn(
                verification.question.family_id,
                {presentation.question.family_id, repair.question.family_id},
                root_id,
            )
            self.engine.submit_answer(
                verification.decision_id,
                verification.question.correct_option.id,
                confidence=0.9,
                response_ms=900,
                idempotency_key=f"all-roots-verify-{index}",
            )

    def test_wrong_answer_enters_targeted_remediation_without_repeating_item(self) -> None:
        session = self.start(mode="diagnose")
        first = self.engine.next_question(session["id"])
        wrong = next(option for option in first.question.options if not option.correct)
        result = self.engine.submit_answer(
            first.decision_id,
            wrong.id,
            confidence=0.8,
            response_ms=1500,
            idempotency_key="answer-1",
        )
        self.assertFalse(result.correct)
        self.assertEqual(result.next_phase, SessionPhase.REMEDIATE)
        self.assertEqual(result.focus_misconception_id, wrong.misconception_id)

        next_presentation = self.engine.next_question(session["id"])
        self.assertNotEqual(next_presentation.question.id, first.question.id)
        self.assertNotEqual(next_presentation.question.family_id, first.question.family_id)

    def test_remediation_success_requires_independent_verification(self) -> None:
        session = self.start()
        first = self.engine.next_question(session["id"])
        wrong = next(option for option in first.question.options if not option.correct)
        self.engine.submit_answer(first.decision_id, wrong.id, idempotency_key="step-1")

        repair = self.engine.next_question(session["id"])
        result = self.engine.submit_answer(
            repair.decision_id, repair.question.correct_option.id, idempotency_key="step-2"
        )
        self.assertTrue(result.correct)
        self.assertEqual(result.next_phase, SessionPhase.VERIFY)

        verification = self.engine.next_question(session["id"])
        self.assertNotIn(
            verification.question.family_id,
            {first.question.family_id, repair.question.family_id},
        )
        verified = self.engine.submit_answer(
            verification.decision_id,
            verification.question.correct_option.id,
            idempotency_key="step-3",
        )
        self.assertEqual(verified.next_phase, SessionPhase.LEARN)
        self.assertIsNone(verified.focus_misconception_id)

    def test_prerequisite_repair_returns_to_original_unresolved_goal(self) -> None:
        session = self.engine.start_session(
            "learner-1", "c_clustering", mode="learn", seed=13
        )
        trigger = self.engine.next_question(session["id"])
        trigger_wrong = next(
            option for option in trigger.question.options if not option.correct
        )
        first_failure = self.engine.submit_answer(
            trigger.decision_id,
            trigger_wrong.id,
            confidence=0.9,
            response_ms=900,
            idempotency_key="parent-trigger",
        )
        original_concept = first_failure.focus_concept_id
        original_misconception = first_failure.focus_misconception_id

        repair = self.engine.next_question(session["id"])
        repair_wrong = next(
            option for option in repair.question.options if not option.correct
        )
        descended = self.engine.submit_answer(
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

        prerequisite_probe = self.engine.next_question(session["id"])
        self.engine.submit_answer(
            prerequisite_probe.decision_id,
            prerequisite_probe.question.correct_option.id,
            confidence=0.9,
            response_ms=900,
            idempotency_key="parent-prerequisite-repair",
        )
        prerequisite_verify = self.engine.next_question(session["id"])
        returned = self.engine.submit_answer(
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

        parent_recheck = self.engine.next_question(session["id"])
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
            )
            first = self.engine.next_question(session["id"])
            uncertain = self.engine.submit_answer(
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
            confirmation = self.engine.next_question(session["id"])
            self.assertNotEqual(
                confirmation.question.family_id, first.question.family_id, label
            )
            confirmed = self.engine.submit_answer(
                confirmation.decision_id,
                confirmation.question.correct_option.id,
                confidence=0.90,
                response_ms=900,
                idempotency_key=f"uncertain-confirm-{index}",
            )
            self.assertEqual(confirmed.next_phase, SessionPhase.LEARN, label)

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
            "learner-1", "c_probability_reasoning", mode="review", seed=17
        )
        first = self.engine.next_question(session["id"])
        wrong = next(option for option in first.question.options if not option.correct)
        self.engine.submit_answer(first.decision_id, wrong.id, idempotency_key="review-1")

        repair = self.engine.next_question(session["id"])
        self.engine.submit_answer(
            repair.decision_id, repair.question.correct_option.id, idempotency_key="review-2"
        )
        self.assertEqual(self.database.get_session(session["id"])["phase"], "verify")

        verify = self.engine.next_question(session["id"])
        verify_wrong = next(option for option in verify.question.options if not option.correct)
        failed = self.engine.submit_answer(
            verify.decision_id, verify_wrong.id, idempotency_key="review-3"
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
                )
                presentation = self.engine.next_question(session["id"])
                key = f"integer-confidence-answer-{confidence}"
                first = self.engine.submit_answer(
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
        presentation = self.engine.next_question(session["id"])
        option_id = presentation.question.correct_option.id
        key = "full-answer-command"
        inputs = {
            "confidence": 0.22,
            "response_ms": 321,
            "hint_count": 1,
            "feedback_shown": False,
        }
        first = self.engine.submit_answer(
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
        presentation = self.engine.next_question(session["id"])
        wrong = next(option for option in presentation.question.options if not option.correct)
        self.engine.submit_answer(presentation.decision_id, wrong.id, idempotency_key="mis-1")
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
        self.assertEqual(response["schema_version"], 1)
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
        presentation = self.engine.next_question(session["id"])
        self.engine.submit_answer(
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
