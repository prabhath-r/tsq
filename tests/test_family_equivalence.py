# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.errors import ConflictError
from tsq.families import (
    canonical_family_label,
    evidence_family_id,
    family_assignment,
    family_alias_members,
)
from tsq.learner import LearnerModel
from tsq.policy import AdaptivePolicy
from tsq.replay import ProjectionReplay
from tsq.store import Database, question_content_hash
from tsq.versions import (
    CANONICAL_FAMILY_V9_MODEL_VERSION,
    OBJECTIVE_GRID_V8_MODEL_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
START = datetime(2104, 8, 9, 9, 0, tzinfo=timezone.utc)


class FamilyEquivalenceTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parsed = read_and_parse(CORPUS, include_catalog=True)
        cls.questions = {question.id: question for question in cls.parsed[4]}

    def test_manifest_is_exact_and_every_alias_resolves_in_the_corpus(self) -> None:
        aliases = family_alias_members()
        self.assertEqual(len(aliases), 49)
        self.assertEqual(set(aliases), {
            question_id
            for question_id, question in self.questions.items()
            if question.published_family_id != question.family_id
        })
        for question_id, assignment in aliases.items():
            question = self.questions[question_id]
            self.assertEqual(
                question.published_family_id,
                assignment.published_family_id,
            )
            self.assertEqual(question.family_id, assignment.evidence_family_id)
            self.assertEqual(
                canonical_family_label(assignment.published_family_id),
                assignment.evidence_family_id,
            )

    def test_an_alias_cannot_be_copied_to_an_unreviewed_question_id(self) -> None:
        question = self.questions["q_attention_permutation_contract_001"]
        with self.assertRaisesRegex(ValueError, "without an explicit"):
            replace(question, id="q_unreviewed_family_alias_fixture_001")

    def test_registry_requires_published_label_but_attempts_accept_canonical(
        self,
    ) -> None:
        question_id = "q_attention_permutation_contract_001"
        assignment = family_alias_members()[question_id]

        with self.assertRaisesRegex(
            ValueError,
            "immutable published family",
        ):
            family_assignment(question_id, assignment.evidence_family_id)
        self.assertEqual(
            evidence_family_id(question_id, assignment.published_family_id),
            assignment.evidence_family_id,
        )
        self.assertEqual(
            evidence_family_id(question_id, assignment.evidence_family_id),
            assignment.evidence_family_id,
        )

    def test_database_preserves_published_identity_but_hydrates_evidence_family(
        self,
    ) -> None:
        question_id = "q_attention_permutation_contract_001"
        expected = self.questions[question_id]
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "family.db")
            database.initialize()
            imported = database.import_corpus(*self.parsed)
            with database.read() as connection:
                raw = connection.execute(
                    "SELECT family_id, content_hash FROM questions WHERE id=?",
                    (question_id,),
                ).fetchone()
                hydrated = database.get_question(
                    question_id,
                    connection,
                    release_id=imported["release_id"],
                )
                sql_family = connection.execute(
                    "SELECT tsq_canonical_family(?) AS family_id",
                    (raw["family_id"],),
                ).fetchone()["family_id"]

        self.assertEqual(raw["family_id"], expected.published_family_id)
        self.assertEqual(raw["content_hash"], question_content_hash(expected))
        self.assertEqual(hydrated.published_family_id, expected.published_family_id)
        self.assertEqual(hydrated.family_id, expected.family_id)
        self.assertEqual(sql_family, expected.family_id)

    def test_canonical_label_cannot_replace_manifested_registry_identity(
        self,
    ) -> None:
        question_id = "q_attention_permutation_contract_001"
        assignment = family_alias_members()[question_id]
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "tampered-family.db")
            database.initialize()
            database.import_corpus(*self.parsed)
            with database.transaction() as connection:
                trigger_sql = connection.execute(
                    """SELECT sql FROM sqlite_master
                       WHERE type='trigger'
                         AND name='questions_immutable_content'"""
                ).fetchone()["sql"]
                connection.execute(
                    "DROP TRIGGER questions_immutable_content"
                )
                connection.execute(
                    "UPDATE questions SET family_id=? WHERE id=?",
                    (assignment.evidence_family_id, question_id),
                )
                connection.execute(trigger_sql)

            with self.assertRaisesRegex(
                ConflictError,
                "invalid question-family registry row.*immutable published",
            ):
                database.validate_current_schema()
            with self.assertRaisesRegex(
                ValueError,
                "immutable published family",
            ):
                database.get_question(question_id)
            integrity = database.verify_integrity()

        self.assertFalse(integrity["ok"])
        self.assertTrue(
            any(
                f"question {question_id}: invalid registry row" in error
                and "immutable published family" in error
                for error in integrity["errors"]
            ),
            integrity["errors"],
        )

    @staticmethod
    def _force_question(
        engine: AdaptiveEngine,
        session_id: str,
        question_id: str,
        *,
        now: datetime,
    ):
        def choose_target(distribution, *, seed, step):
            del seed, step
            for score, probability in distribution:
                if score.question_id == question_id:
                    return score, probability
            raise AssertionError(
                f"Forced question {question_id} was not in the safe frontier."
            )

        original_score = engine.policy._score

        def boost_target(question, **kwargs):
            score = original_score(question, **kwargs)
            return (
                replace(score, total=score.total + 5.0)
                if question.id == question_id
                else score
            )

        with (
            patch.object(
                engine.policy,
                "_score",
                side_effect=boost_target,
            ),
            patch.object(
                AdaptivePolicy,
                "_sample_distribution",
                side_effect=choose_target,
            ),
        ):
            return engine.next_question(session_id, now=now)

    def test_v8_alias_then_v9_canonical_replays_and_rebuilds_exactly(
        self,
    ) -> None:
        alias_id = "q_causal_mask_training_leak_001"
        canonical_id = "q_causal_mask_batch_matrix_001"
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "mixed-family.db")
            database.initialize()
            database.import_corpus(*self.parsed)

            v8 = AdaptiveEngine(
                database,
                LearnerModel(OBJECTIVE_GRID_V8_MODEL_VERSION),
            )
            learner_id = "mixed-family-upgrade"
            v8.create_learner(learner_id)
            first_session = v8.start_session(
                learner_id,
                "c_causal_masking",
                mode="learn",
                seed=18,
                now=START,
            )
            first = self._force_question(
                v8,
                first_session["id"],
                alias_id,
                now=START,
            )
            v8.submit_answer(
                first.decision_id,
                first.question.correct_option.id,
                confidence=0.95,
                response_ms=1_200,
                now=START + timedelta(minutes=1),
            )

            v9 = AdaptiveEngine(
                database,
                LearnerModel(CANONICAL_FAMILY_V9_MODEL_VERSION),
            )
            second_at = START + timedelta(days=8)
            second_session = v9.start_session(
                learner_id,
                "c_causal_masking",
                mode="review",
                seed=19,
                now=second_at,
            )
            second = self._force_question(
                v9,
                second_session["id"],
                canonical_id,
                now=second_at,
            )
            result = v9.submit_answer(
                second.decision_id,
                second.question.correct_option.id,
                confidence=0.95,
                response_ms=1_200,
                now=second_at + timedelta(minutes=1),
            )

            with database.read() as connection:
                attempts = connection.execute(
                    """SELECT family_id FROM attempts
                       WHERE learner_id=? ORDER BY answered_at""",
                    (learner_id,),
                ).fetchall()
                belief = connection.execute(
                    """SELECT evidence_count FROM misconception_beliefs
                       WHERE learner_id=? AND misconception_id=?""",
                    (learner_id, "m_mask_only_inference"),
                ).fetchone()
                objective_families = connection.execute(
                    """SELECT family_id FROM learner_objective_families
                       WHERE learner_id=? AND objective_id=?
                       ORDER BY family_id""",
                    (learner_id, "lo_causal_visibility"),
                ).fetchall()
            self.assertEqual(
                [row["family_id"] for row in attempts],
                [
                    "f_causal_mask_training_leak",
                    "f_causal_mask_batch_matrix",
                ],
            )
            objective_change = next(
                change
                for change in result.state_changes
                if change.get("objective_id") == "lo_causal_visibility"
            )
            self.assertEqual(
                objective_change["family_evidence_power"],
                0.75,
            )
            self.assertEqual(belief["evidence_count"], 1)
            self.assertEqual(
                [row["family_id"] for row in objective_families],
                [
                    "f_causal_mask_batch_matrix",
                    "f_causal_mask_training_leak",
                ],
            )
            integrity = database.verify_integrity()
            self.assertTrue(integrity["ok"], integrity["errors"])

            checked = ProjectionReplay(database).check(learner_id)
            self.assertTrue(checked["ok"], checked["errors"])
            self.assertTrue(checked["source_projection_matches_replay"])
            rebuilt_path = Path(directory) / "mixed-family-rebuilt.db"
            rebuilt_report = ProjectionReplay(database).rebuild_copy(
                learner_id,
                rebuilt_path,
            )
            self.assertTrue(rebuilt_report["ok"])
            rebuilt = Database(rebuilt_path)
            self.assertTrue(
                ProjectionReplay(rebuilt).check(learner_id)["ok"]
            )

    def test_mixed_alias_ledgers_dedupe_coverage_and_operation_kind(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "mixed-ledgers.db")
            database.initialize()
            imported = database.import_corpus(*self.parsed)
            learner_id = "mixed-ledger-reader"
            engine = AdaptiveEngine(database)
            engine.create_learner(learner_id)
            old_at = START.isoformat()
            new_at = (START + timedelta(days=1)).isoformat()
            with database.transaction() as connection:
                for table, target_column, target_id in (
                    (
                        "learner_skill_families",
                        "concept_id",
                        "c_causal_masking",
                    ),
                    (
                        "learner_objective_families",
                        "objective_id",
                        "lo_causal_visibility",
                    ),
                ):
                    for family_id, kind, last_at in (
                        (
                            "f_causal_mask_training_leak",
                            "conceptual",
                            old_at,
                        ),
                        (
                            "f_causal_mask_batch_matrix",
                            "application",
                            new_at,
                        ),
                    ):
                        connection.execute(
                            f"""INSERT INTO {table}(
                                   learner_id, {target_column}, family_id,
                                   kind, first_unguided_correct_at,
                                   last_unguided_correct_at,
                                   delayed_unguided_correct_at
                               ) VALUES (?, ?, ?, ?, ?, ?, NULL)""",
                            (
                                learner_id,
                                target_id,
                                family_id,
                                kind,
                                old_at,
                                last_at,
                            ),
                        )

            counts = engine.policy._successful_retrieval_family_counts(
                learner_id,
                imported["release_id"],
                {
                    ("concept", "c_causal_masking"),
                    ("objective", "lo_causal_visibility"),
                },
            )
            self.assertEqual(
                counts,
                {
                    ("concept", "c_causal_masking"): 1,
                    ("objective", "lo_causal_visibility"): 1,
                },
            )
            profile = engine.profile(
                learner_id,
                root_concept_id="c_causal_masking",
                now=START + timedelta(days=2),
            )
            objective = next(
                row
                for row in profile["learning_objectives"]
                if row["objective_id"] == "lo_causal_visibility"
            )
            self.assertEqual(objective["independent_families"], 1)
            self.assertEqual(objective["operation_kinds"], 1)


if __name__ == "__main__":
    unittest.main()
