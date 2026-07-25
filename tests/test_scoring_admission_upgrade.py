# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tsq.store import SCHEMA_VERSION, Database

from tests.test_performance_replay import (
    build_performance_database,
    performance_source_snapshot,
)
from tests.test_scoring_claim_history_upgrade import restore_pre_shadow_schema


def event_fingerprint(database: Database) -> tuple[tuple[object, ...], ...]:
    with database.read() as connection:
        return tuple(
            tuple(row)
            for row in connection.execute(
                "SELECT * FROM events ORDER BY stream_id, stream_version"
            )
        )


class ScoringAdmissionUpgradeTests(unittest.TestCase):
    def test_exact_v14_upgrade_adds_empty_claims_without_changing_history(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exact-v14.db"
            database, _ = build_performance_database(path)

            # Reconstruct the exact pre-claim v14 boundary from a current
            # database. Productive task/action/evaluation history remains
            # populated; later admission and prospective-shadow structures
            # are removed.
            with database.transaction() as connection:
                restore_pre_shadow_schema(connection)
                connection.execute(
                    "DROP TRIGGER events_respect_performance_scoring_claim"
                )
                connection.execute(
                    "DROP TRIGGER task_evaluations_validate_scoring_claim"
                )
                connection.execute("DROP TABLE performance_scoring_claims")
                connection.execute(
                    "UPDATE meta SET value='14' WHERE key='schema_version'"
                )

            before_events = event_fingerprint(database)
            before_performance = performance_source_snapshot(database)
            with database.read() as connection:
                self.assertIsNone(
                    connection.execute(
                        """SELECT 1 FROM sqlite_master
                           WHERE type='table'
                             AND name='performance_scoring_claims'"""
                    ).fetchone()
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT value FROM meta WHERE key='schema_version'"
                    ).fetchone()["value"],
                    "14",
                )

            database.initialize()

            self.assertEqual(SCHEMA_VERSION, 18)
            self.assertEqual(event_fingerprint(database), before_events)
            after_performance = performance_source_snapshot(database)
            self.assertEqual(
                after_performance["projection"],
                before_performance["projection"],
            )
            self.assertEqual(
                after_performance["events"],
                before_performance["events"],
            )
            with database.read() as connection:
                version = connection.execute(
                    "SELECT value FROM meta WHERE key='schema_version'"
                ).fetchone()["value"]
                claim_count = connection.execute(
                    "SELECT COUNT(*) AS n FROM performance_scoring_claims"
                ).fetchone()["n"]
                evaluation_count = connection.execute(
                    "SELECT COUNT(*) AS n FROM task_evaluations"
                ).fetchone()["n"]
            self.assertEqual(version, "18")
            self.assertEqual(claim_count, 0)
            self.assertEqual(evaluation_count, 1)
            database.validate_current_schema()
            integrity = database.verify_integrity()
            self.assertTrue(integrity["ok"], integrity["errors"])

    def test_fresh_v15_schema_installs_claim_table_and_all_guards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "fresh-v15.db")
            database.initialize()

            with database.read() as connection:
                version = connection.execute(
                    "SELECT value FROM meta WHERE key='schema_version'"
                ).fetchone()["value"]
                table = connection.execute(
                    """SELECT 1 FROM sqlite_master
                       WHERE type='table'
                         AND name='performance_scoring_claims'"""
                ).fetchone()
                triggers = {
                    row["name"]
                    for row in connection.execute(
                        """SELECT name FROM sqlite_master
                           WHERE type='trigger'"""
                    )
                }
            self.assertEqual(version, "18")
            self.assertIsNotNone(table)
            self.assertTrue(
                {
                    "performance_scoring_claims_validate_insert",
                    "performance_scoring_claims_no_update",
                    "performance_scoring_claims_no_delete",
                    "events_respect_performance_scoring_claim",
                    "task_evaluations_validate_scoring_claim",
                }
                <= triggers
            )
            database.validate_current_schema()


if __name__ == "__main__":
    unittest.main()
