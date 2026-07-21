# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.errors import ValidationError
from tsq.models import QuestionStatus
from tsq.store import CANDIDATE_POOL_SQL, Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"


REFERENCE_SQL = """SELECT q.id,
       MIN(CASE
           WHEN qc.concept_id = ? THEN 0
           WHEN EXISTS (
               SELECT 1 FROM options ox
               WHERE ox.question_id = q.id AND ox.misconception_id = ?
           ) THEN 0 ELSE 1
       END) AS focus_rank,
       (SELECT COUNT(*) FROM decisions personal
        JOIN questions presented ON presented.id = personal.question_id
        WHERE personal.learner_id = ? AND presented.family_id = q.family_id
       ) AS personal_exposures,
       ABS(q.difficulty - ?) AS difficulty_distance
FROM questions q
JOIN release_questions rq
  ON rq.question_id = q.id AND rq.release_id = ?
JOIN question_concepts qc ON qc.question_id = q.id
JOIN requested_scope scope ON scope.id = qc.concept_id
WHERE rq.status IN (?, ?) AND qc.role = 'primary'
GROUP BY q.id
ORDER BY focus_rank, personal_exposures, difficulty_distance,
         q.discrimination DESC, q.id
LIMIT ?"""


class CandidateRetrievalTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.tempdir.name) / "candidate.db")
        self.database.initialize()
        self.release_id = self.database.import_corpus(
            *read_and_parse(CORPUS, include_catalog=True)
        )[
            "release_id"
        ]
        self.learner_id = "candidate-learner"
        self.engine = AdaptiveEngine(self.database)
        self.engine.create_learner(self.learner_id, "Candidate learner")
        self.root_concept_id = "c_adaptive_testing"
        self.scope = self.database.get_graph(self.release_id).learning_scope(
            self.root_concept_id
        )

        # Give the learner a non-empty, non-uniform exposure history so the
        # equivalence test covers the personal-exposure ordering term.
        session = self.engine.start_session(
            self.learner_id,
            self.root_concept_id,
            mode="learn",
            seed=193,
        )
        for index in range(3):
            presentation = self.engine.next_question(session["id"])
            self.engine.submit_answer(
                presentation.decision_id,
                presentation.question.correct_option.id,
                idempotency_key=f"candidate-history-{index}",
            )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _reference_ids(
        self,
        *,
        focus_concept_id: str | None,
        focus_misconception_id: str | None,
        target_difficulty: float,
        limit: int,
    ) -> list[str]:
        with self.database.read() as connection:
            connection.execute("CREATE TEMP TABLE requested_scope(id TEXT PRIMARY KEY)")
            connection.executemany(
                "INSERT INTO requested_scope(id) VALUES (?)",
                ((concept_id,) for concept_id in sorted(self.scope)),
            )
            rows = connection.execute(
                REFERENCE_SQL,
                (
                    focus_concept_id,
                    focus_misconception_id,
                    self.learner_id,
                    target_difficulty,
                    self.release_id,
                    QuestionStatus.APPROVED.value,
                    QuestionStatus.CALIBRATED.value,
                    limit,
                ),
            ).fetchall()
        return [row["id"] for row in rows]

    def test_optimized_query_preserves_reference_ordering_semantics(self) -> None:
        seed_questions = self.database.questions_for_scope(
            self.scope,
            learner_id=self.learner_id,
            release_id=self.release_id,
            limit=20,
        )
        focus_question = next(
            question for question in seed_questions if question.misconception_ids
        )
        misconception_id = sorted(focus_question.misconception_ids)[0]
        scenarios = (
            (None, None, 0.0),
            (focus_question.primary_concept_id, None, focus_question.difficulty),
            (None, misconception_id, focus_question.difficulty),
            (
                focus_question.primary_concept_id,
                misconception_id,
                focus_question.difficulty + 0.17,
            ),
            (None, None, -3.5),
        )
        for focus_concept_id, focus_misconception_id, target_difficulty in scenarios:
            with self.subTest(
                focus_concept_id=focus_concept_id,
                focus_misconception_id=focus_misconception_id,
                target_difficulty=target_difficulty,
            ):
                expected = self._reference_ids(
                    focus_concept_id=focus_concept_id,
                    focus_misconception_id=focus_misconception_id,
                    target_difficulty=target_difficulty,
                    limit=17,
                )
                actual = [
                    question.id
                    for question in self.database.questions_for_scope(
                        self.scope,
                        learner_id=self.learner_id,
                        focus_concept_id=focus_concept_id,
                        focus_misconception_id=focus_misconception_id,
                        release_id=self.release_id,
                        target_difficulty=target_difficulty,
                        limit=17,
                    )
                ]
                self.assertEqual(actual, expected)
                # Repetition must not depend on planner or row-production order.
                repeated = [
                    question.id
                    for question in self.database.questions_for_scope(
                        self.scope,
                        learner_id=self.learner_id,
                        focus_concept_id=focus_concept_id,
                        focus_misconception_id=focus_misconception_id,
                        release_id=self.release_id,
                        target_difficulty=target_difficulty,
                        limit=17,
                    )
                ]
                self.assertEqual(repeated, expected)

    def test_production_query_uses_scope_and_focus_covering_indexes(self) -> None:
        focus_question = self.database.questions_for_scope(
            self.scope,
            learner_id=self.learner_id,
            release_id=self.release_id,
            limit=1,
        )[0]
        misconception_id = sorted(focus_question.misconception_ids)[0]
        with self.database.read() as connection:
            connection.execute("CREATE TEMP TABLE requested_scope(id TEXT PRIMARY KEY)")
            connection.executemany(
                "INSERT INTO requested_scope(id) VALUES (?)",
                ((concept_id,) for concept_id in sorted(self.scope)),
            )
            plan = [
                row["detail"]
                for row in connection.execute(
                    "EXPLAIN QUERY PLAN " + CANDIDATE_POOL_SQL,
                    (
                        focus_question.primary_concept_id,
                        misconception_id,
                        focus_question.difficulty,
                        self.release_id,
                        self.learner_id,
                        QuestionStatus.APPROVED.value,
                        QuestionStatus.CALIBRATED.value,
                        17,
                    ),
                ).fetchall()
            ]
            question_concept_indexes = {
                row["name"]
                for row in connection.execute("PRAGMA index_list(question_concepts)")
            }
            option_indexes = {
                row["name"] for row in connection.execute("PRAGMA index_list(options)")
            }

        self.assertIn(
            "idx_question_concepts_primary_scope", question_concept_indexes
        )
        self.assertIn("idx_options_misconception_question", option_indexes)
        self.assertFalse(any(detail == "SCAN qc" for detail in plan), plan)
        self.assertTrue(
            any(
                "SEARCH qc USING COVERING INDEX idx_question_concepts_primary_scope"
                in detail
                for detail in plan
            ),
            plan,
        )
        self.assertTrue(
            any(
                "SEARCH focused USING COVERING INDEX idx_options_misconception_question"
                in detail
                for detail in plan
            ),
            plan,
        )
        self.assertTrue(
            any(
                "SEARCH presented USING COVERING INDEX idx_decisions_learner_question"
                in detail
                for detail in plan
            ),
            plan,
        )

    def test_candidate_hydration_bound_cannot_exceed_sql_kernel_contract(self) -> None:
        for invalid_limit in (0, 601, True):
            with self.subTest(limit=invalid_limit), self.assertRaises(ValidationError):
                self.database.questions_for_scope(
                    self.scope,
                    learner_id=self.learner_id,
                    release_id=self.release_id,
                    limit=invalid_limit,
                )

    def test_prolific_family_cannot_hide_independent_families_beyond_bound(self) -> None:
        database = Database(Path(self.tempdir.name) / "family-diversity.db")
        database.initialize()
        release_id = "release_family_diversity"
        concept_id = "c_family_diversity"
        timestamp = "2026-01-01T00:00:00+00:00"
        with database.transaction() as connection:
            connection.execute(
                """INSERT INTO concepts(
                       id, content_hash, name, description, domain, prior_mastery
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    concept_id,
                    "0" * 64,
                    "Family diversity",
                    "Synthetic concept for bounded candidate retrieval.",
                    "test",
                    0.25,
                ),
            )
            connection.execute(
                "INSERT INTO corpus_releases(id, bundle_hash, created_at) VALUES (?, ?, ?)",
                (release_id, "1" * 64, timestamp),
            )
            connection.execute(
                "INSERT INTO release_concepts(release_id, concept_id) VALUES (?, ?)",
                (release_id, concept_id),
            )
            questions = []
            mappings = []
            options = []
            for index in range(602):
                question_id = f"q_family_diversity_{index:04d}"
                family_id = (
                    "family_a"
                    if index < 600
                    else "family_b" if index == 600 else "family_c"
                )
                questions.append(
                    (
                        question_id,
                        1,
                        f"{index:064x}",
                        family_id,
                        "approved",
                        f"Synthetic family-diversity question {index}?",
                        "application",
                        0.0,
                        1.0,
                        0.25,
                        0.05,
                        "{}",
                        "[]",
                        None,
                        timestamp,
                    )
                )
                mappings.append((question_id, concept_id, 1.0, "primary"))
                options.extend(
                    (
                        (question_id, "a", "Correct option", 1, "Correct.", None),
                        (question_id, "b", "Distractor one", 0, "Incorrect.", None),
                        (question_id, "c", "Distractor two", 0, "Incorrect.", None),
                        (question_id, "d", "Distractor three", 0, "Incorrect.", None),
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
                       question_id, option_id, text, is_correct, rationale,
                       misconception_id
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                options,
            )
            connection.executemany(
                """INSERT INTO release_questions(
                       release_id, question_id, status, evidence_weight
                   ) VALUES (?, ?, 'approved', 0.85)""",
                ((release_id, row[0]) for row in questions),
            )
            connection.execute(
                "UPDATE corpus_releases SET sealed_at = ? WHERE id = ?",
                (timestamp, release_id),
            )
            connection.execute(
                "INSERT INTO meta(key, value) VALUES ('active_corpus_release', ?)",
                (release_id,),
            )

        engine = AdaptiveEngine(database)
        engine.create_learner("family-diversity-learner")
        pool = database.questions_for_scope(
            {concept_id},
            learner_id="family-diversity-learner",
            release_id=release_id,
            limit=600,
        )
        self.assertEqual(len(pool), 600)
        self.assertEqual(
            {question.family_id for question in pool[:3]},
            {"family_a", "family_b", "family_c"},
        )
        session = engine.start_session(
            "family-diversity-learner", concept_id, seed=37
        )
        presentation = engine.next_question(session["id"])
        self.assertIn(
            presentation.question.family_id,
            {"family_a", "family_b", "family_c"},
        )


class BenchmarkFixtureTestCase(unittest.TestCase):
    def test_small_benchmark_bank_is_serviceable_by_the_live_policy(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "src")
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "benchmarks" / "benchmark_candidate_retrieval.py"),
                "--questions",
                "100",
                "--rounds",
                "2",
                "--json",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["question_count"], 100)
        self.assertEqual(len(payload["selected_question_ids"]), 2)
        self.assertGreater(payload["full_policy_selection"]["median_ms"], 0)


if __name__ == "__main__":
    unittest.main()
