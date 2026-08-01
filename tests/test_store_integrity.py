# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.errors import ConflictError, NotFoundError, ValidationError
from tsq.models import (
    Concept,
    ConceptEdge,
    ConceptWeight,
    Misconception,
    Option,
    Question,
    QuestionKind,
    QuestionStatus,
    RelationType,
    Source,
)
from tsq.store import Database, question_content_hash

from tests.schema_upgrade_helpers import restore_pre_shadow_schema


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"


def tiny_corpus(question_status: QuestionStatus = QuestionStatus.APPROVED):
    concepts = (
        Concept("c_base", "Base", "A prerequisite."),
        Concept("c_target", "Target", "The target concept."),
    )
    edges = (
        ConceptEdge("c_base", "c_target", RelationType.PREREQUISITE),
    )
    sources = (
        Source(
            "src_tiny",
            "Tiny test source",
            "https://example.test/tiny",
            "CC0",
            {"locator": "Tiny fixture, section 1"},
        ),
    )
    misconceptions = (
        Misconception(
            "m_replacement",
            "c_target",
            "Replacement confusion",
            "Treats a prerequisite as something the target replaces.",
        ),
        Misconception(
            "m_reversal",
            "c_target",
            "Direction reversal",
            "Reverses the direction of the prerequisite relationship.",
        ),
        Misconception(
            "m_independence",
            "c_target",
            "Independence confusion",
            "Treats the target as independent of its prerequisite.",
        ),
    )
    question = Question(
        id="q_target",
        version=1,
        family_id="family_target",
        status=question_status,
        stem=(
            "Given that Base is a prerequisite of Target, which statement best "
            "describes their relationship?"
        ),
        kind=QuestionKind.CONCEPTUAL,
        difficulty=0.0,
        discrimination=1.0,
        guess_rate=0.25,
        slip_rate=0.05,
        concepts=(ConceptWeight("c_target", 1.0, "primary"),),
        options=(
            Option(
                "correct",
                "The target rule depends on the stated base condition.",
                True,
                "This preserves the directed prerequisite relationship.",
            ),
            Option(
                "wrong",
                "The target rule replaces the stated base condition.",
                False,
                "A prerequisite remains required; the target does not replace it.",
                "m_replacement",
            ),
            Option(
                "wrong_2",
                "The base rule depends on the stated target condition.",
                False,
                "This reverses the direction of the prerequisite relationship.",
                "m_reversal",
            ),
            Option(
                "wrong_3",
                "The target rule is independent of the base condition.",
                False,
                "Independence contradicts the stated prerequisite relationship.",
                "m_independence",
            ),
        ),
        source_ids=("src_tiny",),
        provenance={
            "generated": False,
            "authoring_method": "expert-authored-test-fixture",
        },
    )
    return concepts, edges, misconceptions, sources, (question,)


class StoreIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.tempdir.name) / "test.db")
        self.database.initialize()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_release_graph_and_question_status_are_pinned(self) -> None:
        first_bundle = tiny_corpus()
        first_release = self.database.import_corpus(*first_bundle)["release_id"]
        question = first_bundle[-1][0]

        self.database.ensure_learner("learner")
        session = self.database.create_session("learner", "c_target", seed=7)
        selected_score = {
            "total": 1.0,
            "predicted_correct": 0.5,
            "information_gain": 0.5,
            "learning_fit": 0.5,
            "concept_need": 0.5,
            "misconception_value": 0.0,
            "prerequisite_value": 0.0,
            "review_value": 0.0,
            "novelty": 1.0,
            "kind_fit": 1.0,
        }
        now = datetime.now(timezone.utc).isoformat()
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO decisions(
                       id, session_id, learner_id, question_id, question_version,
                       question_content_hash, question_status, evidence_weight,
                       corpus_release_id, session_revision, learner_revision, phase,
                       focus_concept_id, focus_misconception_id, pedagogical_role,
                       focus_valid, policy_version, candidate_count, candidate_digest,
                       top_candidates_json, selected_score_json, propensity,
                       option_order_json, rationale, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "decision",
                    session["id"],
                    "learner",
                    question.id,
                    question.version,
                    question_content_hash(question),
                    QuestionStatus.APPROVED.value,
                    QuestionStatus.APPROVED.evidence_weight,
                    first_release,
                    0,
                    0,
                    "learn",
                    None,
                    None,
                    "learning",
                    1,
                    "test-policy",
                    1,
                    hashlib.sha256(b"candidate").hexdigest(),
                    "[]",
                    json.dumps(selected_score),
                    1.0,
                    json.dumps(["correct", "wrong", "wrong_2", "wrong_3"]),
                    "test",
                    now,
                ),
            )

        calibrated = replace(question, status=QuestionStatus.CALIBRATED)
        second_release = self.database.import_corpus(
            first_bundle[0], (), first_bundle[2], first_bundle[3], (calibrated,)
        )["release_id"]

        self.assertEqual(len(self.database.get_graph(first_release).edges), 1)
        self.assertEqual(len(self.database.get_graph(second_release).edges), 0)
        self.assertEqual(len(self.database.get_graph().edges), 0)
        with self.assertRaises(NotFoundError):
            self.database.get_graph("missing-release")

        first_questions = self.database.questions_for_scope(
            {"c_target"}, release_id=first_release
        )
        second_questions = self.database.questions_for_scope(
            {"c_target"}, release_id=second_release
        )
        self.assertEqual(first_questions[0].status, QuestionStatus.APPROVED)
        self.assertEqual(second_questions[0].status, QuestionStatus.CALIBRATED)
        self.assertEqual(
            self.database.pending_presentation(session["id"]).question.status,
            QuestionStatus.APPROVED,
        )

    def test_database_import_is_itself_a_quality_trust_boundary(self) -> None:
        bundle = tiny_corpus()
        invalid_approved = replace(
            bundle[-1][0],
            options=bundle[-1][0].options[:2],
        )
        with self.assertRaisesRegex(
            ValidationError, "Corpus activation failed"
        ):
            self.database.import_corpus(
                bundle[0], bundle[1], bundle[2], bundle[3], (invalid_approved,)
            )
        with self.database.read() as connection:
            releases = connection.execute(
                "SELECT COUNT(*) AS n FROM corpus_releases"
            ).fetchone()["n"]
        self.assertEqual(releases, 0)

    def test_numeric_hashes_survive_sqlite_real_round_trip(self) -> None:
        bundle = tiny_corpus()
        integer_numeric_question = replace(
            bundle[-1][0],
            difficulty=0,
            discrimination=1,
            concepts=(ConceptWeight("c_target", 1, "primary"),),
        )
        integer_weight_edge = ConceptEdge(
            "c_base", "c_target", RelationType.PREREQUISITE, 1
        )
        self.database.import_corpus(
            bundle[0],
            (integer_weight_edge,),
            bundle[2],
            bundle[3],
            (integer_numeric_question,),
        )
        report = self.database.verify_integrity()
        self.assertTrue(report["ok"], report["errors"])

    def test_release_manifest_excludes_objects_omitted_by_a_later_bundle(self) -> None:
        bundle = tiny_corpus()
        first_release = self.database.import_corpus(*bundle)["release_id"]
        target_only = (bundle[0][1],)
        second_release = self.database.import_corpus(
            target_only, (), (), (), ()
        )["release_id"]

        self.assertNotEqual(first_release, second_release)
        self.assertEqual(
            set(self.database.get_graph(first_release).concepts),
            {"c_base", "c_target"},
        )
        self.assertEqual(
            set(self.database.get_graph(second_release).concepts), {"c_target"}
        )
        self.assertEqual(
            len(
                self.database.questions_for_scope(
                    {"c_target"}, release_id=first_release
                )
            ),
            1,
        )
        self.assertEqual(
            self.database.questions_for_scope(
                {"c_target"}, release_id=second_release
            ),
            [],
        )
        self.database.initialize()
        self.assertEqual(set(self.database.get_graph().concepts), {"c_target"})

    def test_initialize_installs_family_index_and_attempt_update_guard(self) -> None:
        with self.database.read() as connection:
            indexes = {
                row["name"] for row in connection.execute("PRAGMA index_list(attempts)")
            }
            triggers = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger'"
                )
            }
        self.assertIn("idx_attempts_learner_family", indexes)
        self.assertIn("attempts_no_update", triggers)

    def test_corpus_release_snapshots_are_append_only(self) -> None:
        release_id = self.database.import_corpus(*tiny_corpus())["release_id"]
        forbidden = (
            (
                "UPDATE release_questions SET status = 'retired' WHERE release_id = ?",
                (release_id,),
            ),
            (
                "DELETE FROM release_edges WHERE release_id = ?",
                (release_id,),
            ),
            (
                "UPDATE corpus_releases SET bundle_hash = 'changed' WHERE id = ?",
                (release_id,),
            ),
        )
        for statement, parameters in forbidden:
            with self.subTest(statement=statement), self.assertRaises(sqlite3.IntegrityError):
                with self.database.transaction() as connection:
                    connection.execute(statement, parameters)

        release_membership_inserts = (
            (
                "INSERT INTO release_concepts(release_id, concept_id) VALUES (?, 'c_base')",
                (release_id,),
            ),
            (
                """INSERT INTO release_edges(
                       release_id, source_id, target_id, relation, weight
                   ) VALUES (?, 'c_base', 'c_target', 'prerequisite', 1.0)""",
                (release_id,),
            ),
            (
                """INSERT INTO release_misconceptions(release_id, misconception_id)
                   VALUES (?, 'm_replacement')""",
                (release_id,),
            ),
            (
                "INSERT INTO release_sources(release_id, source_id) VALUES (?, 'src_tiny')",
                (release_id,),
            ),
            (
                """INSERT INTO release_questions(
                       release_id, question_id, status, evidence_weight
                   ) VALUES (?, 'q_target', 'approved', 1.0)""",
                (release_id,),
            ),
        )
        for statement, parameters in release_membership_inserts:
            with self.subTest(statement=statement), self.assertRaisesRegex(
                sqlite3.IntegrityError,
                "sealed corpus releases cannot gain members",
            ):
                with self.database.transaction() as connection:
                    connection.execute(statement, parameters)

        released_component_inserts = (
            """INSERT INTO question_concepts(question_id, concept_id, weight, role)
               VALUES ('q_target', 'c_base', 0.1, 'secondary')""",
            """INSERT INTO options(
                   question_id, option_id, text, is_correct, rationale, misconception_id
               ) VALUES (
                   'q_target', 'late_option', 'A post-release option', 0,
                   'This component was added after release.', 'm_replacement'
               )""",
            """INSERT INTO question_sources(question_id, source_id)
               VALUES ('q_target', 'src_tiny')""",
        )
        for statement in released_component_inserts:
            with self.subTest(statement=statement), self.assertRaisesRegex(
                sqlite3.IntegrityError,
                "released question components are sealed",
            ):
                with self.database.transaction() as connection:
                    connection.execute(statement)

        with self.assertRaisesRegex(
            sqlite3.IntegrityError, "active corpus release must be sealed"
        ):
            with self.database.transaction() as connection:
                connection.execute(
                    """INSERT INTO corpus_releases(id, bundle_hash, created_at)
                       VALUES ('unsealed', 'unsealed-hash', ?)""",
                    (datetime.now(timezone.utc).isoformat(),),
                )
                connection.execute(
                    """UPDATE meta SET value = 'unsealed'
                       WHERE key = 'active_corpus_release'"""
                )

    def test_catalog_snapshot_and_cross_topic_projection_are_release_pinned(self) -> None:
        parsed = read_and_parse(CORPUS, include_catalog=True)
        release_id = self.database.import_corpus(*parsed)["release_id"]

        catalog = self.database.get_catalog(release_id)
        self.assertEqual(len(catalog["domains"]), 1)
        self.assertEqual(len(catalog["topics"]), 16)
        self.assertEqual(
            {
                entry["relation"]
                for entry in self.database.question_topics(
                    "q_rag_grounding_boundaries_001", release_id
                )
            },
            {"primary", "cross"},
        )
        self.assertTrue(
            {
                "c_attention",
                "c_transformers",
                "c_attention_scaling",
                "c_causal_masking",
            }.issubset(self.database.topic_scope("t_transformers", release_id))
        )
        with self.assertRaises(sqlite3.IntegrityError):
            with self.database.transaction() as connection:
                connection.execute(
                    """UPDATE release_topics SET name = 'Changed'
                       WHERE release_id = ? AND topic_id = 't_transformers'""",
                    (release_id,),
                )

    def test_integrity_reconstructs_catalog_hashes_after_trigger_bypass(self) -> None:
        release_id = self.database.import_corpus(
            *read_and_parse(CORPUS, include_catalog=True)
        )["release_id"]
        with self.database.transaction() as connection:
            self.database._drop_release_snapshot_triggers(connection)
            connection.execute(
                """UPDATE release_topics
                   SET description = 'tampered', content_hash = 'forged'
                   WHERE release_id = ? AND topic_id = 't_transformers'""",
                (release_id,),
            )

        report = self.database.verify_integrity()

        self.assertFalse(report["ok"])
        self.assertTrue(
            any(
                "topic t_transformers content hash mismatch" in error
                for error in report["errors"]
            ),
            report["errors"],
        )
        self.assertTrue(
            any("bundle hash mismatch" in error for error in report["errors"]),
            report["errors"],
        )

    def test_v6_migration_adds_catalog_tables_without_rehashing_legacy_release(self) -> None:
        legacy = Database(Path(self.tempdir.name) / "legacy-v6.db")
        legacy.initialize()
        release_id = legacy.import_corpus(*read_and_parse(CORPUS))["release_id"]
        with legacy.transaction() as connection:
            restore_pre_shadow_schema(connection)
            legacy._drop_release_snapshot_triggers(connection)
            for table in (
                "release_question_topics",
                "release_topic_concepts",
                "release_topics",
                "release_domains",
            ):
                connection.execute(f"DROP TABLE {table}")
            connection.execute("ALTER TABLE sessions DROP COLUMN topic_id")
            connection.execute("ALTER TABLE sessions DROP COLUMN exploration_mode")
            connection.execute(
                "UPDATE meta SET value = '6' WHERE key = 'schema_version'"
            )

        legacy.initialize()

        with legacy.read() as connection:
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(sessions)")
            }
            tables = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            stored_release = connection.execute(
                "SELECT id FROM corpus_releases WHERE id = ?", (release_id,)
            ).fetchone()
        self.assertTrue({"topic_id", "exploration_mode"}.issubset(columns))
        self.assertTrue(
            {
                "release_domains",
                "release_topics",
                "release_topic_concepts",
                "release_question_topics",
            }.issubset(tables)
        )
        self.assertIsNotNone(stored_release)
        self.assertEqual(legacy.get_catalog(release_id)["topics"], [])
        self.assertTrue(legacy.verify_integrity()["ok"])

    def test_versioned_corpus_registry_is_immutable_except_question_status(self) -> None:
        bundle = tiny_corpus()
        self.database.import_corpus(*bundle)
        forbidden = (
            "UPDATE concepts SET name = 'silently changed' WHERE id = 'c_target'",
            "UPDATE questions SET stem = 'silently changed' WHERE id = 'q_target'",
            "UPDATE options SET text = 'silently changed' WHERE question_id = 'q_target'",
            "DELETE FROM question_concepts WHERE question_id = 'q_target'",
        )
        for statement in forbidden:
            with self.subTest(statement=statement), self.assertRaises(sqlite3.IntegrityError):
                with self.database.transaction() as connection:
                    connection.execute(statement)

        # Lifecycle movement is intentionally mutable in the registry.  Sessions
        # still use their release-pinned status and evidence weight.
        calibrated = replace(bundle[-1][0], status=QuestionStatus.CALIBRATED)
        self.database.import_corpus(
            bundle[0], bundle[1], bundle[2], bundle[3], (calibrated,)
        )
        with self.database.read() as connection:
            status = connection.execute(
                "SELECT status FROM questions WHERE id = 'q_target'"
            ).fetchone()["status"]
        self.assertEqual(status, QuestionStatus.CALIBRATED.value)

    def test_integrity_recomputes_registry_and_release_manifest_hashes(self) -> None:
        bundle = tiny_corpus()
        release_id = self.database.import_corpus(*bundle)["release_id"]
        tampered_question = replace(bundle[-1][0], stem="A coordinated rewrite.")
        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER questions_immutable_content")
            connection.execute(
                "UPDATE questions SET stem = ?, content_hash = ? WHERE id = ?",
                (
                    tampered_question.stem,
                    question_content_hash(tampered_question),
                    tampered_question.id,
                ),
            )

        corrupted = self.database.verify_integrity()
        self.assertFalse(corrupted["ok"])
        self.assertTrue(
            any(
                error == f"release {release_id}: bundle hash mismatch"
                for error in corrupted["errors"]
            ),
            corrupted["errors"],
        )

    def test_integrity_rejects_cross_release_question_dependencies(self) -> None:
        release_id = self.database.import_corpus(*tiny_corpus())["release_id"]
        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER release_concepts_no_delete")
            connection.execute(
                "DELETE FROM release_concepts WHERE release_id = ? AND concept_id = ?",
                (release_id, "c_target"),
            )

        corrupted = self.database.verify_integrity()
        self.assertFalse(corrupted["ok"])
        self.assertTrue(
            any(
                "question q_target maps concept c_target outside the release" in error
                for error in corrupted["errors"]
            ),
            corrupted["errors"],
        )

    def test_future_schema_is_rejected_before_ddl(self) -> None:
        path = Path(self.tempdir.name) / "future.db"
        connection = sqlite3.connect(path)
        connection.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT INTO meta VALUES('schema_version', '999')")
        connection.commit()
        connection.close()

        future = Database(path)
        with self.assertRaises(ConflictError):
            future.initialize()
        connection = sqlite3.connect(path)
        names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        connection.close()
        self.assertEqual(names, {"meta"})

    def test_integrity_checks_full_envelope_and_stream_head(self) -> None:
        self.database.import_corpus(*tiny_corpus())
        clean = self.database.verify_integrity()
        self.assertTrue(clean["ok"], clean["errors"])
        self.assertEqual(clean["quick_check"], ["ok"])

        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER events_no_update")
            connection.execute(
                """UPDATE events SET occurred_at = '1900-01-01T00:00:00+00:00'
                   WHERE event_type = 'CorpusImported'"""
            )
        tampered = self.database.verify_integrity()
        self.assertFalse(tampered["ok"])
        self.assertTrue(
            any("payload hash mismatch" in error for error in tampered["errors"])
        )

        other = Database(Path(self.tempdir.name) / "tail.db")
        other.initialize()
        with other.transaction() as connection:
            other.append_event(
                connection,
                stream_id="standalone",
                event_type="Standalone",
                payload={"value": 1},
            )
            connection.execute("DROP TRIGGER events_no_delete")
            connection.execute("DELETE FROM events WHERE stream_id = 'standalone'")
        truncated = other.verify_integrity("standalone")
        self.assertFalse(truncated["ok"])
        self.assertTrue(any("head" in error for error in truncated["errors"]))
        with self.assertRaises(ConflictError):
            with other.transaction() as connection:
                other.append_event(
                    connection,
                    stream_id="standalone",
                    event_type="WouldMaskTruncation",
                    payload={"value": 2},
                )

    def test_integrity_rejects_non_finite_event_json_without_crashing(self) -> None:
        with self.database.transaction() as connection:
            self.database.append_event(
                connection,
                stream_id="payload-corruption",
                event_type="Probe",
                payload={"value": 1},
            )
            self.database.append_event(
                connection,
                stream_id="metadata-corruption",
                event_type="Probe",
                payload={"value": 2},
                metadata={"source": "test"},
            )
            self.database.append_event(
                connection,
                stream_id="duplicate-key-corruption",
                event_type="Probe",
                payload={"value": 3},
            )
            self.database.append_event(
                connection,
                stream_id="overflow-number-corruption",
                event_type="Probe",
                payload={"value": 4},
            )
            connection.execute("DROP TRIGGER events_no_update")
            connection.execute(
                "UPDATE events SET payload_json = ? WHERE stream_id = ?",
                ('{"value":NaN}', "payload-corruption"),
            )
            connection.execute(
                "UPDATE events SET metadata_json = ? WHERE stream_id = ?",
                ('{"value":Infinity}', "metadata-corruption"),
            )
            connection.execute(
                "UPDATE events SET payload_json = ? WHERE stream_id = ?",
                ('{"value":1,"value":2}', "duplicate-key-corruption"),
            )
            connection.execute(
                "UPDATE events SET payload_json = ? WHERE stream_id = ?",
                ('{"value":1e999}', "overflow-number-corruption"),
            )

        corrupted = self.database.verify_integrity()

        self.assertFalse(corrupted["ok"])
        self.assertTrue(
            any(
                "invalid payload JSON" in error
                and "non-finite JSON constant NaN" in error
                for error in corrupted["errors"]
            ),
            corrupted["errors"],
        )
        self.assertTrue(
            any(
                "invalid payload JSON" in error
                and "non-finite JSON number 1e999" in error
                for error in corrupted["errors"]
            ),
            corrupted["errors"],
        )
        self.assertTrue(
            any(
                "invalid metadata JSON" in error
                and "non-finite JSON constant Infinity" in error
                for error in corrupted["errors"]
            ),
            corrupted["errors"],
        )
        self.assertTrue(
            any(
                "invalid payload JSON" in error
                and "duplicate JSON object key 'value'" in error
                for error in corrupted["errors"]
            ),
            corrupted["errors"],
        )

    def test_integrity_rejects_non_finite_attempt_confidence_without_crashing(
        self,
    ) -> None:
        self.database.import_corpus(
            *read_and_parse(CORPUS, include_catalog=True)
        )
        engine = AdaptiveEngine(self.database)
        engine.create_learner("non-finite-confidence")
        started_at = datetime(2100, 1, 1, tzinfo=timezone.utc)
        session = engine.start_session(
            "non-finite-confidence",
            "t_machine_learning",
            seed=11,
            now=started_at,
        )
        presentation = engine.next_question(session["id"], now=started_at)
        engine.submit_answer(
            presentation.decision_id,
            presentation.question.correct_option.id,
            confidence=0.7,
            response_ms=800,
            now=started_at + timedelta(milliseconds=800),
        )

        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER attempts_no_update")
            connection.execute(
                "UPDATE attempts SET confidence=?",
                (float("inf"),),
            )

        corrupted = self.database.verify_integrity()
        self.assertFalse(corrupted["ok"])
        self.assertTrue(
            any(
                "confidence is out of bounds" in error
                for error in corrupted["errors"]
            ),
            corrupted["errors"],
        )

    def test_integrity_checks_decision_response_and_attempt_projection(self) -> None:
        self.database.import_corpus(
            *read_and_parse(CORPUS, include_catalog=True)
        )
        engine = AdaptiveEngine(self.database)
        engine.create_learner("learner")
        started_at = datetime(2100, 1, 1, tzinfo=timezone.utc)
        session = engine.start_session(
            "learner", "t_machine_learning", seed=11, now=started_at
        )
        presentation = engine.next_question(session["id"], now=started_at)
        engine.submit_answer(
            presentation.decision_id,
            presentation.question.correct_option.id,
            confidence=0.7,
            response_ms=800,
            idempotency_key="answer",
            now=started_at + timedelta(milliseconds=800),
        )
        clean = self.database.verify_integrity()
        self.assertTrue(clean["ok"], clean["errors"])

        with self.assertRaises(sqlite3.IntegrityError):
            with self.database.transaction() as connection:
                connection.execute("UPDATE attempts SET confidence = 0.2")

        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER attempts_no_update")
            connection.execute(
                "UPDATE attempts SET confidence = 0.2, outcome_json = '{}'"
            )
        corrupted = self.database.verify_integrity()
        self.assertFalse(corrupted["ok"])
        self.assertTrue(
            any(
                "command hash mismatch" in error or "response event" in error
                for error in corrupted["errors"]
            )
        )
        self.assertTrue(any("outcome" in error for error in corrupted["errors"]))


if __name__ == "__main__":
    unittest.main()
