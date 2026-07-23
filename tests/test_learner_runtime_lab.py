# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from experiments.learner_runtime_lab import (
    MAX_HISTORY_RESPONSES,
    PerformanceLabError,
    clone_database,
    database_family_fingerprint,
    file_fingerprint,
    latency_summary,
    run_lab,
)


class LearnerRuntimeLabTests(unittest.TestCase):
    def test_latency_summary_uses_nearest_rank_p95(self) -> None:
        samples = [value * 1_000_000 for value in range(1, 21)]

        summary = latency_summary(samples)

        self.assertEqual(summary["sample_count"], 20)
        self.assertEqual(summary["median_ms"], 10.5)
        self.assertEqual(summary["p95_ms"], 19.0)
        self.assertEqual(summary["min_ms"], 1.0)
        self.assertEqual(summary["max_ms"], 20.0)

    def test_clone_is_consistent_and_does_not_mutate_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "protected.db"
            clone = Path(directory) / "clone.db"
            connection = sqlite3.connect(source)
            try:
                connection.execute("CREATE TABLE sample(value TEXT NOT NULL)")
                connection.execute("INSERT INTO sample(value) VALUES ('stable')")
                connection.commit()
            finally:
                connection.close()
            before = file_fingerprint(source)

            clone_database(source, clone)

            after = file_fingerprint(source)
            self.assertEqual(after, before)
            clone_connection = sqlite3.connect(clone)
            try:
                value = clone_connection.execute(
                    "SELECT value FROM sample"
                ).fetchone()[0]
            finally:
                clone_connection.close()
            self.assertEqual(value, "stable")

    def test_configuration_is_bounded_before_database_work(self) -> None:
        with self.assertRaisesRegex(PerformanceLabError, "history_responses"):
            run_lab(history_responses=MAX_HISTORY_RESPONSES + 1)

    def test_database_family_fingerprint_includes_both_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "protected.db"
            wal = Path(f"{database}-wal")
            shm = Path(f"{database}-shm")
            database.write_bytes(b"database")
            wal.write_bytes(b"write-ahead log")
            shm.write_bytes(b"shared memory")

            before = database_family_fingerprint(database)

            self.assertEqual(set(before), {"database", "wal", "shm"})
            self.assertTrue(all(item["exists"] for item in before.values()))
            self.assertEqual(before["wal"]["path"], str(wal.resolve()))
            self.assertEqual(before["shm"]["path"], str(shm.resolve()))
            wal.write_bytes(b"mutated write-ahead log")
            self.assertNotEqual(database_family_fingerprint(database), before)


if __name__ == "__main__":
    unittest.main()
