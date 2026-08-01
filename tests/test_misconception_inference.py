# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.errors import ConflictError, ValidationError
from tsq.inference import (
    LEGACY_MISCONCEPTION_ALGORITHM,
    MISCONCEPTION_ALGORITHM_METADATA_KEY,
    MISCONCEPTION_ALGORITHM_VERSION,
)
from tsq.learner import (
    MAX_MISCONCEPTION_FAMILY_CONTRIBUTION,
    MODEL_VERSION,
    OBJECTIVE_GRID_V6_MODEL_VERSION,
    LearnerModel,
)
from tsq.models import (
    ConceptRole,
    ConceptWeight,
    Option,
    Question,
    QuestionKind,
    QuestionStatus,
    logit,
)
from tsq.replay import ProjectionReplay
from tsq.store import Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
NOW = datetime(2103, 2, 3, 9, 0, tzinfo=timezone.utc)
MISCONCEPTION_ID = "m_target"
CONCEPT_ID = "c_target"


def reducer_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE events (
            event_id TEXT PRIMARY KEY,
            stream_id TEXT NOT NULL,
            stream_version INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            occurred_at TEXT NOT NULL,
            learner_id TEXT,
            metadata_json TEXT NOT NULL
        );
        CREATE TABLE attempts (
            event_id TEXT PRIMARY KEY,
            learner_id TEXT NOT NULL,
            question_id TEXT NOT NULL,
            family_id TEXT NOT NULL,
            selected_option_id TEXT,
            is_correct INTEGER NOT NULL,
            confidence REAL,
            response_ms INTEGER,
            hint_count INTEGER NOT NULL
        );
        CREATE TABLE questions (
            id TEXT PRIMARY KEY,
            discrimination REAL NOT NULL
        );
        CREATE TABLE options (
            question_id TEXT NOT NULL,
            option_id TEXT NOT NULL,
            misconception_id TEXT,
            PRIMARY KEY (question_id, option_id)
        );
        CREATE TABLE question_concepts (
            question_id TEXT NOT NULL,
            concept_id TEXT NOT NULL,
            role TEXT NOT NULL
        );
        CREATE TABLE misconceptions (
            id TEXT PRIMARY KEY,
            concept_id TEXT NOT NULL
        );
        CREATE TABLE misconception_beliefs (
            learner_id TEXT NOT NULL,
            misconception_id TEXT NOT NULL,
            log_odds REAL NOT NULL,
            evidence_count INTEGER NOT NULL,
            last_seen_at TEXT NOT NULL,
            as_of_event_id TEXT NOT NULL,
            model_version TEXT NOT NULL,
            PRIMARY KEY (learner_id, misconception_id)
        );
        """
    )
    connection.execute(
        "INSERT INTO misconceptions(id, concept_id) VALUES (?, ?)",
        (MISCONCEPTION_ID, CONCEPT_ID),
    )
    return connection


def question_for(question_id: str, family_id: str) -> Question:
    return Question(
        id=question_id,
        version=1,
        family_id=family_id,
        status=QuestionStatus.CALIBRATED,
        stem=(
            "Which response applies the target distinction under the stated "
            "conditions?"
        ),
        kind=QuestionKind.APPLICATION,
        difficulty=0.0,
        discrimination=1.5,
        guess_rate=0.25,
        slip_rate=0.05,
        concepts=(
            ConceptWeight(CONCEPT_ID, 1.0, ConceptRole.PRIMARY),
        ),
        options=(
            Option(
                "correct",
                "The keyed distinction applies.",
                True,
                "This applies the distinction.",
            ),
            Option(
                "target",
                "The named misconception applies.",
                False,
                "This instantiates the named misconception.",
                MISCONCEPTION_ID,
            ),
            Option(
                "other",
                "An unrelated error applies.",
                False,
                "This is a different error.",
            ),
            Option(
                "premise",
                "An unstated premise decides it.",
                False,
                "This invents a premise.",
            ),
        ),
        source_ids=("s_test",),
    )


class ReducerHarness:
    def __init__(self, model_version: str = MODEL_VERSION) -> None:
        self.connection = reducer_connection()
        self.model = LearnerModel(model_version)
        self.model_version = model_version
        self.version = 0

    def close(self) -> None:
        self.connection.close()

    def append(
        self,
        question: Question,
        *,
        correct: bool,
        confidence: float | None = 0.95,
        response_ms: int | None = 1_000,
        hint_count: int = 0,
        algorithm: str = MISCONCEPTION_ALGORITHM_VERSION,
        apply: bool = True,
    ) -> str:
        self.connection.execute(
            """INSERT OR IGNORE INTO questions(id, discrimination)
               VALUES (?, ?)""",
            (question.id, question.discrimination),
        )
        self.connection.execute(
            """INSERT OR IGNORE INTO question_concepts(
                   question_id, concept_id, role
               ) VALUES (?, ?, 'primary')""",
            (question.id, CONCEPT_ID),
        )
        for option in question.options:
            self.connection.execute(
                """INSERT OR IGNORE INTO options(
                       question_id, option_id, misconception_id
                   ) VALUES (?, ?, ?)""",
                (question.id, option.id, option.misconception_id),
            )
        self.version += 1
        event_id = f"evt_{self.version:04d}"
        metadata = {
            "learner_model_version": self.model_version,
            "evidence_weight": 1.0,
        }
        if algorithm != LEGACY_MISCONCEPTION_ALGORITHM:
            metadata[MISCONCEPTION_ALGORITHM_METADATA_KEY] = algorithm
        self.connection.execute(
            """INSERT INTO events(
                   event_id, stream_id, stream_version, event_type,
                   occurred_at, learner_id, metadata_json
               ) VALUES (?, 'learner:learner', ?, 'ResponseSubmitted',
                         ?, 'learner', ?)""",
            (
                event_id,
                self.version,
                NOW.isoformat(),
                json.dumps(metadata, sort_keys=True),
            ),
        )
        selected = (
            question.correct_option
            if correct
            else next(
                option
                for option in question.options
                if option.misconception_id == MISCONCEPTION_ID
            )
        )
        self.connection.execute(
            """INSERT INTO attempts(
                   event_id, learner_id, question_id, family_id,
                   selected_option_id, is_correct, confidence, response_ms,
                   hint_count
               ) VALUES (?, 'learner', ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                question.id,
                question.family_id,
                selected.id,
                int(correct),
                confidence,
                response_ms,
                hint_count,
            ),
        )
        if apply:
            prior_family_attempts = self.connection.execute(
                """SELECT COUNT(*) AS n FROM attempts
                   WHERE learner_id='learner' AND family_id=?
                     AND event_id<>?""",
                (question.family_id, event_id),
            ).fetchone()["n"]
            legacy_weight = (
                question.status.evidence_weight
                * self.model.family_dependence_discount(
                    prior_family_attempts
                )
            )
            self.model._update_misconceptions(
                self.connection,
                learner_id="learner",
                question=question,
                selected_option=selected,
                event_id=event_id,
                now=NOW,
                evidence_weight=legacy_weight,
                confidence=confidence,
                misconception_algorithm=algorithm,
            )
        return event_id

    @property
    def log_odds(self) -> float:
        row = self.connection.execute(
            """SELECT log_odds FROM misconception_beliefs
               WHERE learner_id='learner' AND misconception_id=?""",
            (MISCONCEPTION_ID,),
        ).fetchone()
        return logit(0.10) if row is None else row["log_odds"]

    @property
    def evidence_count(self) -> int:
        row = self.connection.execute(
            """SELECT evidence_count FROM misconception_beliefs
               WHERE learner_id='learner' AND misconception_id=?""",
            (MISCONCEPTION_ID,),
        ).fetchone()
        return 0 if row is None else row["evidence_count"]


class ReversibleMisconceptionReducerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = ReducerHarness()
        self.question = question_for("q_same", "f_same")

    def tearDown(self) -> None:
        self.harness.close()

    def test_same_family_latest_credible_sign_replaces_prior_sign(self) -> None:
        positive = LearnerModel.misconception_family_contribution(
            model_version=MODEL_VERSION,
            selected_misconception_id=MISCONCEPTION_ID,
            misconception_id=MISCONCEPTION_ID,
            correct=False,
            confidence=0.95,
            response_ms=1_000,
            hint_count=0,
            base_evidence_weight=1.0,
            discrimination=1.5,
        )
        negative = LearnerModel.misconception_family_contribution(
            model_version=MODEL_VERSION,
            selected_misconception_id=None,
            misconception_id=MISCONCEPTION_ID,
            correct=True,
            confidence=0.95,
            response_ms=1_000,
            hint_count=0,
            base_evidence_weight=1.0,
            discrimination=1.5,
        )
        assert positive is not None and negative is not None

        self.harness.append(self.question, correct=False)
        first_positive = self.harness.log_odds
        self.assertAlmostEqual(first_positive, logit(0.10) + positive)

        self.harness.append(self.question, correct=True)
        self.assertAlmostEqual(
            self.harness.log_odds, logit(0.10) + negative
        )

        self.harness.append(self.question, correct=False)
        self.assertAlmostEqual(self.harness.log_odds, first_positive)

        for _ in range(50):
            self.harness.append(self.question, correct=False)
        self.assertAlmostEqual(self.harness.log_odds, first_positive)
        self.assertEqual(self.harness.evidence_count, 1)
        self.assertLessEqual(
            abs(self.harness.log_odds - logit(0.10)),
            MAX_MISCONCEPTION_FAMILY_CONTRIBUTION,
        )

    def test_independent_families_accumulate_but_one_family_does_not(self) -> None:
        first = question_for("q_first", "f_first")
        second = question_for("q_second", "f_second")
        self.harness.append(first, correct=False)
        one_family = self.harness.log_odds
        self.harness.append(second, correct=False)
        two_families = self.harness.log_odds
        self.assertGreater(two_families, one_family)

        for _ in range(20):
            self.harness.append(first, correct=False)
        self.assertAlmostEqual(self.harness.log_odds, two_families)

    def test_noncredible_correct_and_wrong_do_not_replace_family(self) -> None:
        self.harness.append(self.question, correct=False)
        credible_positive = self.harness.log_odds
        for values in (
            {"response_ms": 100},
            {"hint_count": 1},
            {"confidence": 0.20},
            {"confidence": None},
        ):
            self.harness.append(self.question, correct=True, **values)
            self.assertAlmostEqual(
                self.harness.log_odds, credible_positive
            )

        self.harness.append(self.question, correct=True)
        credible_negative = self.harness.log_odds
        self.assertLess(credible_negative, logit(0.10))
        for values in (
            {"response_ms": 100},
            {"hint_count": 1},
            {"confidence": 0.60},
            {"confidence": None},
        ):
            self.harness.append(self.question, correct=False, **values)
            self.assertAlmostEqual(
                self.harness.log_odds, credible_negative
            )

    def test_first_marked_observation_replaces_effective_legacy_family(self) -> None:
        self.harness.append(
            self.question,
            correct=False,
            algorithm=LEGACY_MISCONCEPTION_ALGORITHM,
        )
        legacy_positive = self.harness.log_odds
        self.assertGreater(legacy_positive, logit(0.10))

        self.harness.append(self.question, correct=True)
        expected = LearnerModel.misconception_family_contribution(
            model_version=MODEL_VERSION,
            selected_misconception_id=None,
            misconception_id=MISCONCEPTION_ID,
            correct=True,
            confidence=0.95,
            response_ms=1_000,
            hint_count=0,
            base_evidence_weight=1.0,
            discrimination=1.5,
        )
        assert expected is not None
        self.assertAlmostEqual(
            self.harness.log_odds, logit(0.10) + expected
        )
        self.assertEqual(self.harness.evidence_count, 1)

    def test_reconstruction_excludes_future_attempts_by_stream_version(self) -> None:
        first_event = self.harness.append(self.question, correct=False)
        positive = self.harness.log_odds
        self.harness.append(self.question, correct=True, apply=False)

        reconstructed = (
            self.harness.model._reconstruct_misconception_belief(
                self.harness.connection,
                learner_id="learner",
                misconception_id=MISCONCEPTION_ID,
                through_event_id=first_event,
            )
        )

        self.assertAlmostEqual(reconstructed["log_odds"], positive)
        self.assertTrue(reconstructed["current_observation_applied"])

    def test_reversal_uses_unclamped_family_sum_after_saturation(self) -> None:
        questions = [
            question_for(f"q_sat_{index}", f"f_sat_{index}")
            for index in range(8)
        ]
        for question in questions:
            self.harness.append(question, correct=False)
        self.assertEqual(self.harness.log_odds, 6.0)

        self.harness.append(questions[0], correct=True)
        positive = LearnerModel.misconception_family_contribution(
            model_version=MODEL_VERSION,
            selected_misconception_id=MISCONCEPTION_ID,
            misconception_id=MISCONCEPTION_ID,
            correct=False,
            confidence=0.95,
            response_ms=1_000,
            hint_count=0,
            base_evidence_weight=1.0,
            discrimination=1.5,
        )
        negative = LearnerModel.misconception_family_contribution(
            model_version=MODEL_VERSION,
            selected_misconception_id=None,
            misconception_id=MISCONCEPTION_ID,
            correct=True,
            confidence=0.95,
            response_ms=1_000,
            hint_count=0,
            base_evidence_weight=1.0,
            discrimination=1.5,
        )
        assert positive is not None and negative is not None
        expected_raw = logit(0.10) + 7 * positive + negative
        self.assertAlmostEqual(
            self.harness.log_odds,
            max(-6.0, min(6.0, expected_raw)),
        )

    def test_current_reducer_preserves_the_event_model_telemetry_contract(
        self,
    ) -> None:
        v6 = ReducerHarness(OBJECTIVE_GRID_V6_MODEL_VERSION)
        try:
            v6.append(
                self.question,
                correct=False,
                confidence=0.50,
                response_ms=900,
            )
            self.assertGreater(v6.log_odds, logit(0.10))
            self.assertEqual(v6.evidence_count, 1)

            v6.append(
                self.question,
                correct=True,
                confidence=None,
                response_ms=900,
            )
            self.assertLess(v6.log_odds, logit(0.10))
            self.assertEqual(v6.evidence_count, 1)
        finally:
            v6.close()

        v7 = ReducerHarness(MODEL_VERSION)
        try:
            v7.append(
                self.question,
                correct=False,
                confidence=0.50,
                response_ms=900,
            )
            self.assertAlmostEqual(v7.log_odds, logit(0.10))
            self.assertEqual(v7.evidence_count, 0)
        finally:
            v7.close()


class MisconceptionEventIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.tempdir.name) / "events.db")
        self.database.initialize()
        self.database.import_corpus(
            *read_and_parse(CORPUS, include_catalog=True)
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def submit_one(
        self,
        engine: AdaptiveEngine,
        learner_id: str,
        *,
        seed: int,
        now: datetime,
        correct: bool,
    ) -> None:
        session = engine.start_session(
            learner_id,
            "c_causal_masking",
            seed=seed,
            now=now,
        )
        presentation = engine.next_question(
            session["id"], now=now + timedelta(minutes=1)
        )
        option = (
            presentation.question.correct_option
            if correct
            else next(
                item
                for item in presentation.question.options
                if not item.correct
            )
        )
        engine.submit_answer(
            presentation.decision_id,
            option.id,
            confidence=0.95,
            response_ms=2_000,
            now=now + timedelta(minutes=2),
        )

    def test_new_events_pin_marker_in_both_metadata_objects(self) -> None:
        engine = AdaptiveEngine(self.database)
        engine.create_learner("marked")
        self.submit_one(
            engine,
            "marked",
            seed=17,
            now=NOW,
            correct=False,
        )
        with self.database.read() as connection:
            rows = connection.execute(
                """SELECT event_type, schema_version, metadata_json
                   FROM events
                   WHERE learner_id='marked'
                     AND event_type IN (
                         'ResponseSubmitted', 'LearnerProjectionAdvanced'
                     )
                   ORDER BY stream_version"""
            ).fetchall()
        self.assertEqual(len(rows), 2)
        response, projection = rows
        self.assertEqual(response["schema_version"], 2)
        for row in (response, projection):
            self.assertEqual(
                json.loads(row["metadata_json"])[
                    MISCONCEPTION_ALGORITHM_METADATA_KEY
                ],
                MISCONCEPTION_ALGORITHM_VERSION,
            )
        self.assertTrue(self.database.verify_integrity()["ok"])
        self.assertTrue(ProjectionReplay(self.database).check("marked")["ok"])

    def test_unmarked_legacy_then_marked_current_history_replays_exactly(
        self,
    ) -> None:
        legacy = AdaptiveEngine(
            self.database,
            misconception_algorithm=LEGACY_MISCONCEPTION_ALGORITHM,
        )
        legacy.create_learner("mixed")
        self.submit_one(
            legacy,
            "mixed",
            seed=31,
            now=NOW,
            correct=False,
        )
        current = AdaptiveEngine(self.database)
        self.submit_one(
            current,
            "mixed",
            seed=32,
            now=NOW + timedelta(days=45),
            correct=True,
        )
        with self.database.read() as connection:
            responses = connection.execute(
                """SELECT schema_version, metadata_json FROM events
                   WHERE learner_id='mixed'
                     AND event_type='ResponseSubmitted'
                   ORDER BY stream_version"""
            ).fetchall()
        self.assertEqual(
            [row["schema_version"] for row in responses], [1, 2]
        )
        self.assertNotIn(
            MISCONCEPTION_ALGORITHM_METADATA_KEY,
            json.loads(responses[0]["metadata_json"]),
        )
        self.assertEqual(
            json.loads(responses[1]["metadata_json"])[
                MISCONCEPTION_ALGORITHM_METADATA_KEY
            ],
            MISCONCEPTION_ALGORITHM_VERSION,
        )
        report = ProjectionReplay(self.database).check("mixed")
        self.assertTrue(report["ok"], report["errors"])
        self.assertTrue(report["source_projection_matches_replay"])
        self.assertTrue(report["commitment_matches_replay"])

    def test_runtime_refuses_marker_regression(self) -> None:
        current = AdaptiveEngine(self.database)
        current.create_learner("no-regression")
        self.submit_one(
            current,
            "no-regression",
            seed=41,
            now=NOW,
            correct=True,
        )
        legacy = AdaptiveEngine(
            self.database,
            misconception_algorithm=LEGACY_MISCONCEPTION_ALGORITHM,
        )
        session = legacy.start_session(
            "no-regression",
            "c_causal_masking",
            seed=42,
            now=NOW + timedelta(days=1),
        )
        presentation = legacy.next_question(
            session["id"], now=NOW + timedelta(days=1, minutes=1)
        )
        with self.assertRaisesRegex(ConflictError, "cannot regress"):
            legacy.submit_answer(
                presentation.decision_id,
                presentation.question.correct_option.id,
                confidence=0.95,
                response_ms=2_000,
                now=NOW + timedelta(days=1, minutes=2),
            )

    def test_integrity_rejects_response_projection_marker_mismatch(self) -> None:
        engine = AdaptiveEngine(self.database)
        engine.create_learner("tampered")
        self.submit_one(
            engine,
            "tampered",
            seed=51,
            now=NOW,
            correct=True,
        )
        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER events_no_update")
            row = connection.execute(
                """SELECT event_id, metadata_json FROM events
                   WHERE learner_id='tampered'
                     AND event_type='LearnerProjectionAdvanced'"""
            ).fetchone()
            metadata = json.loads(row["metadata_json"])
            metadata.pop(MISCONCEPTION_ALGORITHM_METADATA_KEY)
            connection.execute(
                "UPDATE events SET metadata_json=? WHERE event_id=?",
                (json.dumps(metadata, sort_keys=True), row["event_id"]),
            )

        integrity = self.database.verify_integrity()
        self.assertFalse(integrity["ok"])
        self.assertTrue(
            any(
                "projection misconception algorithm does not match" in error
                for error in integrity["errors"]
            ),
            integrity["errors"],
        )
        with self.assertRaisesRegex(
            ValidationError, "metadata.*missing.*misconception_algorithm"
        ):
            ProjectionReplay(self.database).check("tampered")

    def test_replay_and_integrity_reject_marker_regression(self) -> None:
        engine = AdaptiveEngine(self.database)
        engine.create_learner("regressed")
        self.submit_one(
            engine,
            "regressed",
            seed=61,
            now=NOW,
            correct=False,
        )
        self.submit_one(
            engine,
            "regressed",
            seed=62,
            now=NOW + timedelta(days=1),
            correct=True,
        )
        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER events_no_update")
            response = connection.execute(
                """SELECT event_id, metadata_json FROM events
                   WHERE learner_id='regressed'
                     AND event_type='ResponseSubmitted'
                   ORDER BY stream_version DESC LIMIT 1"""
            ).fetchone()
            projection = connection.execute(
                """SELECT event_id, metadata_json FROM events
                   WHERE learner_id='regressed'
                     AND event_type='LearnerProjectionAdvanced'
                     AND causation_id=?""",
                (response["event_id"],),
            ).fetchone()
            for row, response_schema in (
                (response, True),
                (projection, False),
            ):
                metadata = json.loads(row["metadata_json"])
                metadata.pop(MISCONCEPTION_ALGORITHM_METADATA_KEY)
                if response_schema:
                    connection.execute(
                        """UPDATE events
                           SET metadata_json=?, schema_version=1
                           WHERE event_id=?""",
                        (
                            json.dumps(metadata, sort_keys=True),
                            row["event_id"],
                        ),
                    )
                else:
                    connection.execute(
                        "UPDATE events SET metadata_json=? WHERE event_id=?",
                        (
                            json.dumps(metadata, sort_keys=True),
                            row["event_id"],
                        ),
                    )

        integrity = self.database.verify_integrity()
        self.assertFalse(integrity["ok"])
        self.assertTrue(
            any(
                "marker regressed to legacy" in error
                for error in integrity["errors"]
            ),
            integrity["errors"],
        )
        with self.assertRaisesRegex(
            ValidationError, "omits the misconception algorithm"
        ):
            ProjectionReplay(self.database).check("regressed")


if __name__ == "__main__":
    unittest.main()
