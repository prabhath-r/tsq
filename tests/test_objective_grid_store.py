# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import tsq.objective_posterior as posterior_module
from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.errors import ValidationError
from tsq.learner import (
    MODEL_VERSION,
    OBJECTIVE_GAUSSIAN_MODEL_VERSION,
    LearnerModel,
)
from tsq.objective_posterior import (
    OBJECTIVE_POSTERIOR_ALGORITHM,
    OBJECTIVE_POSTERIOR_CODEC,
    OBJECTIVE_POSTERIOR_GRID_ID,
    OBJECTIVE_POSTERIOR_SCHEMA_VERSION,
    OBJECTIVE_POSTERIOR_V1_IDENTITY,
    ObjectivePosterior,
    decode_objective_posterior,
)
from tsq.replay import ProjectionReplay
from tsq.store import Database, SCHEMA_VERSION


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"
START = datetime(2101, 1, 2, 9, 0, tzinfo=timezone.utc)


class ObjectiveGridStoreTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.tempdir.name) / "grid.db")
        self.database.initialize()
        self.database.import_corpus(
            *read_and_parse(CORPUS, include_catalog=True)
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def answer_objective(
        self,
        learner_id: str,
        *,
        model_version: str = MODEL_VERSION,
        offset: timedelta = timedelta(),
    ) -> str:
        engine = AdaptiveEngine(
            self.database, LearnerModel(model_version)
        )
        with self.database.read() as connection:
            exists = connection.execute(
                "SELECT 1 FROM learners WHERE id = ?", (learner_id,)
            ).fetchone()
        if exists is None:
            engine.create_learner(learner_id)
        session = engine.start_session(
            learner_id, "t_transformers", mode="learn", seed=0
        )
        presentation = engine.next_question(
            session["id"], now=START + offset
        )
        self.assertIsNotNone(presentation.question.objective_id)
        engine.submit_answer(
            presentation.decision_id,
            presentation.question.correct_option.id,
            confidence=0.9,
            response_ms=1300,
            idempotency_key=f"{learner_id}-{model_version}-{offset.total_seconds()}",
            now=START + offset + timedelta(minutes=1),
        )
        engine.end_session(
            session["id"],
            status="completed",
            reason="objective_grid_test",
        )
        return presentation.question.objective_id

    def seed_v5_parent(self, learner_id: str, presentation) -> str:
        objective = presentation.question.objective
        self.assertIsNotNone(objective)
        state = LearnerModel(
            OBJECTIVE_GAUSSIAN_MODEL_VERSION
        ).initial_objective_state(learner_id, objective)
        seeded_at = START
        with self.database.transaction() as connection:
            seed_event = self.database.append_event(
                connection,
                stream_id=f"learner:{learner_id}",
                event_type="ObjectiveV5SeededForTest",
                payload={"objective_id": objective.id},
                metadata={"learner_model_version": OBJECTIVE_GAUSSIAN_MODEL_VERSION},
                learner_id=learner_id,
                session_id=presentation.session_id,
                occurred_at=seeded_at,
            )
            connection.execute(
                """INSERT INTO objective_states(
                       learner_id, objective_id, mean, variance,
                       stability_hours, exposures, last_seen_at,
                       next_review_at, evidence_mass, as_of_event_id,
                       model_version
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    learner_id,
                    objective.id,
                    state.mean,
                    state.variance,
                    state.stability_hours,
                    state.exposures,
                    seeded_at.isoformat(),
                    None,
                    state.evidence_mass,
                    seed_event["event_id"],
                    OBJECTIVE_GAUSSIAN_MODEL_VERSION,
                ),
            )
        return seed_event["event_id"]

    def test_schema_twelve_persists_and_hashes_exact_posterior(self) -> None:
        objective_id = self.answer_objective("grid-persist")

        with self.database.read() as connection:
            schema_version = connection.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()["value"]
            row = connection.execute(
                """SELECT * FROM objective_grid_states
                   WHERE learner_id = ? AND objective_id = ?""",
                ("grid-persist", objective_id),
            ).fetchone()
            event = connection.execute(
                """SELECT schema_version, payload_json FROM events
                   WHERE learner_id = ?
                     AND event_type = 'LearnerProjectionAdvanced'
                   ORDER BY stream_version DESC LIMIT 1""",
                ("grid-persist",),
            ).fetchone()

        self.assertEqual(schema_version, str(SCHEMA_VERSION))
        self.assertEqual(SCHEMA_VERSION, 13)
        self.assertEqual(row["posterior_schema_version"], OBJECTIVE_POSTERIOR_SCHEMA_VERSION)
        self.assertEqual(row["algorithm"], OBJECTIVE_POSTERIOR_ALGORITHM)
        self.assertEqual(row["grid_id"], OBJECTIVE_POSTERIOR_GRID_ID)
        self.assertEqual(row["codec"], OBJECTIVE_POSTERIOR_CODEC)
        self.assertEqual(
            hashlib.sha256(row["posterior_blob"]).hexdigest(),
            row["posterior_sha256"],
        )
        state = self.database.get_objective_states("grid-persist")[objective_id]
        self.assertIsNotNone(state.posterior)
        self.assertEqual(state.model_version, MODEL_VERSION)
        self.assertEqual(event["schema_version"], 4)
        payload = json.loads(event["payload_json"])
        self.assertEqual(payload["projection_hash_version"], 3)
        self.assertEqual(
            payload["projection_hash"],
            self.database.learner_projection_hash(
                "grid-persist", hash_version=3
            ),
        )
        integrity = self.database.verify_integrity()
        self.assertTrue(integrity["ok"], integrity["errors"])

    def test_schema_v1_identity_is_frozen_across_future_default_aliases(
        self,
    ) -> None:
        old_objective_id = self.answer_objective("grid-identity-old")
        with self.database.read() as connection:
            old_row = connection.execute(
                """SELECT * FROM objective_grid_states
                   WHERE learner_id = ? AND objective_id = ?""",
                ("grid-identity-old", old_objective_id),
            ).fetchone()
            old_blob = old_row["posterior_blob"]
            old_digest = old_row["posterior_sha256"]

        future_aliases = {
            "OBJECTIVE_POSTERIOR_SCHEMA_VERSION": 2,
            "OBJECTIVE_POSTERIOR_ALGORITHM": "objective-density-grid-v2",
            "OBJECTIVE_POSTERIOR_GRID_ID": "theta[-24,24]/64-v2",
            "OBJECTIVE_POSTERIOR_CODEC": "compact-binary-v2",
        }
        with patch.multiple(posterior_module, **future_aliases):
            decoded = decode_objective_posterior(
                old_blob, expected_digest=old_digest
            )
            self.assertIsInstance(decoded, ObjectivePosterior)

            new_blob = ObjectivePosterior.from_prior(0.20).encode()
            new_payload = json.loads(new_blob)
            self.assertEqual(
                {
                    "schema_version": new_payload["schema_version"],
                    "algorithm": new_payload["algorithm"],
                    "grid_id": new_payload["grid_id"],
                    "codec": new_payload["codec"],
                },
                {
                    "schema_version": OBJECTIVE_POSTERIOR_V1_IDENTITY.schema_version,
                    "algorithm": OBJECTIVE_POSTERIOR_V1_IDENTITY.algorithm,
                    "grid_id": OBJECTIVE_POSTERIOR_V1_IDENTITY.grid_id,
                    "codec": OBJECTIVE_POSTERIOR_V1_IDENTITY.codec,
                },
            )

            new_objective_id = self.answer_objective(
                "grid-identity-new", offset=timedelta(days=1)
            )
            with self.database.read() as connection:
                new_row = connection.execute(
                    """SELECT * FROM objective_grid_states
                       WHERE learner_id = ? AND objective_id = ?""",
                    ("grid-identity-new", new_objective_id),
                ).fetchone()
            self.assertEqual(
                (
                    new_row["posterior_schema_version"],
                    new_row["algorithm"],
                    new_row["grid_id"],
                    new_row["codec"],
                ),
                (
                    OBJECTIVE_POSTERIOR_V1_IDENTITY.schema_version,
                    OBJECTIVE_POSTERIOR_V1_IDENTITY.algorithm,
                    OBJECTIVE_POSTERIOR_V1_IDENTITY.grid_id,
                    OBJECTIVE_POSTERIOR_V1_IDENTITY.codec,
                ),
            )
            self.assertIsNotNone(
                self.database.get_objective_states("grid-identity-old")[
                    old_objective_id
                ].posterior
            )
            self.assertIsNotNone(
                self.database.get_objective_states("grid-identity-new")[
                    new_objective_id
                ].posterior
            )

    def test_exact_state_loader_and_integrity_fail_closed_on_tampering(self) -> None:
        objective_id = self.answer_objective("grid-tamper")
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE objective_grid_states
                   SET mastery_probability = mastery_probability + 0.01
                   WHERE learner_id = ? AND objective_id = ?""",
                ("grid-tamper", objective_id),
            )

        with self.assertRaisesRegex(
            ValidationError, "stored mastery probability"
        ):
            self.database.get_objective_states("grid-tamper")
        report = self.database.verify_integrity()
        self.assertFalse(report["ok"])
        self.assertTrue(
            any(
                "stored mastery probability" in error
                for error in report["errors"]
            ),
            report["errors"],
        )

    def test_replay_repairs_missing_exact_child_on_a_database_copy(self) -> None:
        objective_id = self.answer_objective("grid-repair")
        with self.database.transaction() as connection:
            connection.execute(
                """DELETE FROM objective_grid_states
                   WHERE learner_id = ? AND objective_id = ?""",
                ("grid-repair", objective_id),
            )

        with self.assertRaisesRegex(ValidationError, "missing its exact posterior"):
            self.database.get_objective_states("grid-repair")
        checked = ProjectionReplay(self.database).check("grid-repair")
        self.assertFalse(checked["ok"])
        self.assertTrue(checked["rebuild_safe"], checked["errors"])
        self.assertTrue(checked["commitment_matches_replay"])
        self.assertIsNone(checked["source_projection_hash"])

        rebuilt_path = Path(self.tempdir.name) / "rebuilt.db"
        rebuilt_report = ProjectionReplay(self.database).rebuild_copy(
            "grid-repair", rebuilt_path
        )
        self.assertTrue(rebuilt_report["ok"], rebuilt_report["errors"])
        rebuilt = Database(rebuilt_path)
        state = rebuilt.get_objective_states("grid-repair")[objective_id]
        self.assertIsNotNone(state.posterior)
        integrity = rebuilt.verify_integrity()
        self.assertTrue(integrity["ok"], integrity["errors"])

    def test_v11_migration_keeps_v5_parent_only_and_hash_bytes(self) -> None:
        objective_id = self.answer_objective(
            "grid-v5", model_version=OBJECTIVE_GAUSSIAN_MODEL_VERSION
        )
        before_hash = self.database.learner_projection_hash(
            "grid-v5", hash_version=2
        )
        with self.database.transaction() as connection:
            child_count = connection.execute(
                """SELECT COUNT(*) AS n FROM objective_grid_states
                   WHERE learner_id = ? AND objective_id = ?""",
                ("grid-v5", objective_id),
            ).fetchone()["n"]
            self.assertEqual(child_count, 0)
            connection.execute("DROP TABLE objective_grid_states")
            connection.execute(
                "UPDATE meta SET value='11' WHERE key='schema_version'"
            )

        self.database.initialize()

        state = self.database.get_objective_states("grid-v5")[objective_id]
        self.assertIsNone(state.posterior)
        self.assertEqual(
            state.model_version, OBJECTIVE_GAUSSIAN_MODEL_VERSION
        )
        self.assertEqual(
            before_hash,
            self.database.learner_projection_hash("grid-v5", hash_version=2),
        )
        with self.database.read() as connection:
            child_count = connection.execute(
                "SELECT COUNT(*) AS n FROM objective_grid_states"
            ).fetchone()["n"]
        self.assertEqual(child_count, 0)
        integrity = self.database.verify_integrity()
        self.assertTrue(integrity["ok"], integrity["errors"])

    def test_mixed_v5_to_v6_history_replays_with_declared_versions(self) -> None:
        first_objective = self.answer_objective(
            "grid-mixed", model_version=OBJECTIVE_GAUSSIAN_MODEL_VERSION
        )
        second_objective = self.answer_objective(
            "grid-mixed", offset=timedelta(days=2)
        )

        with self.database.read() as connection:
            events = connection.execute(
                """SELECT schema_version, payload_json, metadata_json
                   FROM events WHERE learner_id = ?
                     AND event_type = 'LearnerProjectionAdvanced'
                   ORDER BY stream_version""",
                ("grid-mixed",),
            ).fetchall()
        self.assertEqual([row["schema_version"] for row in events], [3, 4])
        self.assertEqual(
            [json.loads(row["payload_json"])["projection_hash_version"] for row in events],
            [2, 3],
        )
        self.assertEqual(
            [json.loads(row["metadata_json"])["learner_model_version"] for row in events],
            [OBJECTIVE_GAUSSIAN_MODEL_VERSION, MODEL_VERSION],
        )
        states = self.database.get_objective_states("grid-mixed")
        self.assertIn(first_objective, states)
        self.assertIn(second_objective, states)
        self.assertTrue(
            any(state.posterior is not None for state in states.values())
        )
        replay = ProjectionReplay(self.database).check("grid-mixed")
        self.assertTrue(replay["ok"], replay["errors"])
        self.assertEqual(replay["learner_model_version"], MODEL_VERSION)
        self.assertTrue(replay["source_projection_matches_replay"])
        self.assertTrue(replay["commitment_matches_replay"])

    def test_first_v6_response_converts_same_v5_parent_once_and_retry_is_inert(
        self,
    ) -> None:
        learner_id = "grid-convert-once"
        engine = AdaptiveEngine(self.database)
        engine.create_learner(learner_id)
        session = engine.start_session(
            learner_id, "t_transformers", mode="learn", seed=0
        )
        presentation = engine.next_question(session["id"], now=START)
        objective = presentation.question.objective
        self.assertIsNotNone(objective)
        seed_event_id = self.seed_v5_parent(learner_id, presentation)
        seeded = self.database.get_objective_states(learner_id)[objective.id]
        self.assertEqual(
            seeded.model_version, OBJECTIVE_GAUSSIAN_MODEL_VERSION
        )
        self.assertIsNone(seeded.posterior)

        result = engine.submit_answer(
            presentation.decision_id,
            presentation.question.correct_option.id,
            confidence=0.9,
            response_ms=1200,
            idempotency_key="convert-v5-once",
            now=START + timedelta(minutes=1),
        )
        self.assertFalse(result.idempotent_replay)
        with self.database.read() as connection:
            response_event_id = connection.execute(
                "SELECT event_id FROM attempts WHERE decision_id = ?",
                (presentation.decision_id,),
            ).fetchone()["event_id"]
            parent = connection.execute(
                """SELECT * FROM objective_states
                   WHERE learner_id = ? AND objective_id = ?""",
                (learner_id, objective.id),
            ).fetchone()
            child_row = connection.execute(
                """SELECT * FROM objective_grid_states
                   WHERE learner_id = ? AND objective_id = ?""",
                (learner_id, objective.id),
            ).fetchone()
            child_before_retry = tuple(child_row)
            child_values = dict(child_row)
        self.assertNotEqual(response_event_id, seed_event_id)
        self.assertEqual(parent["as_of_event_id"], response_event_id)
        self.assertEqual(parent["model_version"], MODEL_VERSION)
        self.assertEqual(child_values["as_of_event_id"], response_event_id)
        self.assertEqual(child_values["model_version"], MODEL_VERSION)
        converted = self.database.get_objective_states(learner_id)[objective.id]
        self.assertIsNotNone(converted.posterior)
        self.assertEqual(
            [
                observation.observation_id
                for observation in converted.posterior.pending_observations
            ],
            [response_event_id],
        )

        retry = engine.submit_answer(
            presentation.decision_id,
            presentation.question.correct_option.id,
            confidence=0.9,
            response_ms=1200,
            idempotency_key="convert-v5-once",
            now=START + timedelta(minutes=1),
        )
        self.assertTrue(retry.idempotent_replay)
        with self.database.read() as connection:
            child_after_retry = tuple(
                connection.execute(
                    """SELECT * FROM objective_grid_states
                       WHERE learner_id = ? AND objective_id = ?""",
                    (learner_id, objective.id),
                ).fetchone()
            )
            self.assertEqual(
                connection.execute(
                    """SELECT exposures FROM objective_states
                       WHERE learner_id = ? AND objective_id = ?""",
                    (learner_id, objective.id),
                ).fetchone()["exposures"],
                1,
            )
        self.assertEqual(child_after_retry, child_before_retry)
        with self.assertRaisesRegex(ValueError, "cannot be downgraded"):
            LearnerModel(
                OBJECTIVE_GAUSSIAN_MODEL_VERSION
            ).project_objective_state(
                converted,
                objective,
                START + timedelta(days=1),
            )

    def test_grid_write_failure_rolls_back_parent_and_response(self) -> None:
        learner_id = "grid-atomic-failure"
        engine = AdaptiveEngine(self.database)
        engine.create_learner(learner_id)
        session = engine.start_session(
            learner_id, "t_transformers", mode="learn", seed=0
        )
        presentation = engine.next_question(session["id"], now=START)
        objective = presentation.question.objective
        self.assertIsNotNone(objective)
        self.seed_v5_parent(learner_id, presentation)
        with self.database.read() as connection:
            parent_before = tuple(
                connection.execute(
                    """SELECT * FROM objective_states
                       WHERE learner_id = ? AND objective_id = ?""",
                    (learner_id, objective.id),
                ).fetchone()
            )

        with patch.object(
            Database,
            "upsert_objective_grid_state",
            side_effect=RuntimeError("forced exact-posterior write failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "forced exact-posterior"):
                engine.submit_answer(
                    presentation.decision_id,
                    presentation.question.correct_option.id,
                    confidence=0.9,
                    response_ms=1200,
                    idempotency_key="forced-grid-failure",
                    now=START + timedelta(minutes=1),
                )

        with self.database.read() as connection:
            parent_after = tuple(
                connection.execute(
                    """SELECT * FROM objective_states
                       WHERE learner_id = ? AND objective_id = ?""",
                    (learner_id, objective.id),
                ).fetchone()
            )
            child_count = connection.execute(
                """SELECT COUNT(*) AS n FROM objective_grid_states
                   WHERE learner_id = ? AND objective_id = ?""",
                (learner_id, objective.id),
            ).fetchone()["n"]
            attempt_count = connection.execute(
                "SELECT COUNT(*) AS n FROM attempts WHERE decision_id = ?",
                (presentation.decision_id,),
            ).fetchone()["n"]
            decision = connection.execute(
                "SELECT consumed_at FROM decisions WHERE id = ?",
                (presentation.decision_id,),
            ).fetchone()
            learner_revision = connection.execute(
                "SELECT revision FROM learners WHERE id = ?",
                (learner_id,),
            ).fetchone()["revision"]
            response_event_count = connection.execute(
                """SELECT COUNT(*) AS n FROM events
                   WHERE learner_id = ? AND event_type = 'ResponseSubmitted'""",
                (learner_id,),
            ).fetchone()["n"]
        self.assertEqual(parent_after, parent_before)
        self.assertEqual(child_count, 0)
        self.assertEqual(attempt_count, 0)
        self.assertIsNone(decision["consumed_at"])
        self.assertEqual(learner_revision, 0)
        self.assertEqual(response_event_count, 0)


if __name__ == "__main__":
    unittest.main()
