# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.store import SCHEMA_VERSION, Database

from tests.schema_upgrade_helpers import restore_pre_shadow_schema


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"
V14_PERFORMANCE_TABLES = {
    "performance_tasks",
    "performance_task_releases",
    "release_performance_tasks",
    "performance_attempts",
    "performance_actions",
    "task_evaluations",
    "shadow_evidence_bundles",
}


def event_fingerprint(database: Database) -> tuple[tuple[object, ...], ...]:
    with database.read() as connection:
        return tuple(
            tuple(row)
            for row in connection.execute(
                "SELECT * FROM events ORDER BY stream_id, stream_version"
            )
        )


def learner_fingerprint(database: Database) -> str:
    with database.read() as connection:
        material = []
        for table in (
            "learners",
            "sessions",
            "skill_states",
            "objective_states",
            "objective_grid_states",
            "misconception_beliefs",
            "learner_skill_families",
            "learner_objective_families",
        ):
            rows = connection.execute(
                f'SELECT * FROM "{table}" ORDER BY rowid'
            ).fetchall()
            material.append((table, [tuple(row) for row in rows]))
    return hashlib.sha256(repr(material).encode("utf-8")).hexdigest()


class ProductiveLedgerUpgradeTests(unittest.TestCase):
    def test_v13_upgrade_adds_an_empty_shadow_ledger_without_fabricating_history(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-v13.db"
            database = Database(path)
            database.initialize()
            database.import_corpus(
                *read_and_parse(CORPUS, include_catalog=True)
            )
            engine = AdaptiveEngine(database)
            engine.create_learner("v14-learner", "V14 Learner")
            engine.start_session(
                "v14-learner", "t_transformers", seed=1401
            )

            with database.transaction() as connection:
                restore_pre_shadow_schema(connection)
                connection.execute(
                    "DROP TRIGGER events_respect_performance_scoring_claim"
                )
                connection.execute(
                    "DROP TRIGGER task_evaluations_validate_scoring_claim"
                )
                for table in (
                    "shadow_evidence_bundles",
                    "task_evaluations",
                    "performance_actions",
                    "performance_scoring_claims",
                    "performance_attempts",
                    "release_performance_tasks",
                    "performance_task_releases",
                    "performance_tasks",
                ):
                    connection.execute(f'DROP TABLE "{table}"')
                connection.execute(
                    "UPDATE meta SET value='13' WHERE key='schema_version'"
                )

            before_events = event_fingerprint(database)
            before_learner = learner_fingerprint(database)
            database.initialize()

            self.assertEqual(SCHEMA_VERSION, 21)
            self.assertEqual(event_fingerprint(database), before_events)
            self.assertEqual(learner_fingerprint(database), before_learner)
            with database.read() as connection:
                tables = {
                    row["name"]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                version = connection.execute(
                    "SELECT value FROM meta WHERE key='schema_version'"
                ).fetchone()["value"]
                counts = {
                    table: connection.execute(
                        f'SELECT COUNT(*) AS n FROM "{table}"'
                    ).fetchone()["n"]
                    for table in V14_PERFORMANCE_TABLES
                }
                scoring_claim_count = connection.execute(
                    """SELECT COUNT(*) AS n
                       FROM performance_scoring_claims"""
                ).fetchone()["n"]

            self.assertEqual(version, str(SCHEMA_VERSION))
            self.assertTrue(V14_PERFORMANCE_TABLES <= tables)
            self.assertEqual(set(counts.values()), {0})
            self.assertEqual(scoring_claim_count, 0)
            database.validate_current_schema()
            self.assertTrue(database.verify_integrity()["ok"])


if __name__ == "__main__":
    unittest.main()
