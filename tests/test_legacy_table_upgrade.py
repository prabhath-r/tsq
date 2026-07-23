# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import hashlib
import io
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stderr, redirect_stdout
from pathlib import Path

from tsq.cli import main
from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.errors import ConflictError
from tsq.store import SCHEMA_VERSION, Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"


def durable_database_fingerprint(path: Path) -> tuple[tuple[str, int, str], ...]:
    result: list[tuple[str, int, str]] = []
    for suffix in ("", "-wal", "-journal"):
        candidate = Path(f"{path}{suffix}")
        if candidate.exists() and candidate.stat().st_size:
            material = candidate.read_bytes()
            result.append(
                (
                    suffix or "main",
                    len(material),
                    hashlib.sha256(material).hexdigest(),
                )
            )
    return tuple(result)


class LegacyTableUpgradeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "legacy-v12.db"
        self.database = Database(self.path)
        self.database.initialize()
        self.database.import_corpus(
            *read_and_parse(CORPUS, include_catalog=True)
        )
        AdaptiveEngine(self.database).create_learner(
            "legacy-learner",
            "Legacy Learner",
        )
        with self.database.read() as connection:
            self.question = connection.execute(
                """SELECT question.id, question.content_hash,
                          question.family_id, question.kind,
                          mapping.concept_id
                   FROM questions question
                   JOIN question_concepts mapping
                     ON mapping.question_id = question.id
                    AND mapping.role = 'primary'
                   ORDER BY question.id LIMIT 1"""
            ).fetchone()
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO learner_skill_families(
                       learner_id, concept_id, family_id, kind,
                       first_unguided_correct_at,
                       last_unguided_correct_at,
                       delayed_unguided_correct_at
                   ) VALUES(?, ?, ?, ?, ?, ?, ?)""",
                (
                    "legacy-learner",
                    self.question["concept_id"],
                    self.question["family_id"],
                    self.question["kind"],
                    "2020-01-01T00:00:00+00:00",
                    "2020-01-02T00:00:00+00:00",
                    "2020-01-03T00:00:00+00:00",
                ),
            )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def make_nullable_v12_fixture(self) -> None:
        with self.database.transaction() as connection:
            hashes = connection.execute(
                "SELECT id, content_hash FROM questions ORDER BY id"
            ).fetchall()
            self.database._drop_corpus_registry_triggers(connection)
            connection.execute(
                "ALTER TABLE questions DROP COLUMN content_hash"
            )
            connection.execute(
                "ALTER TABLE questions ADD COLUMN content_hash TEXT"
            )
            connection.executemany(
                """UPDATE questions SET content_hash = ?
                   WHERE id = ?""",
                (
                    (row["content_hash"], row["id"])
                    for row in hashes
                    if row["id"] != self.question["id"]
                ),
            )
            connection.execute(
                "ALTER TABLE learner_skill_families DROP COLUMN kind"
            )
            connection.execute(
                "ALTER TABLE learner_skill_families ADD COLUMN kind TEXT"
            )
            connection.execute(
                """UPDATE meta SET value = '12'
                   WHERE key = 'schema_version'"""
            )

    def test_cli_requires_explicit_migration_then_preserves_legacy_data(
        self,
    ) -> None:
        self.make_nullable_v12_fixture()

        before = durable_database_fingerprint(self.path)
        error = io.StringIO()
        with redirect_stderr(error):
            exit_code = main(
                ["--db", str(self.path), "topics", "--json"]
            )
        self.assertEqual(exit_code, 2)
        self.assertIn("current schema 13", error.getvalue())
        self.assertEqual(durable_database_fingerprint(self.path), before)

        output = io.StringIO()
        error = io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            exit_code = main(
                [
                    "--db",
                    str(self.path),
                    "init",
                    "--corpus",
                    str(CORPUS),
                    "--json",
                ]
            )
        self.assertEqual(exit_code, 0, error.getvalue())
        with closing(sqlite3.connect(self.path)) as connection:
            schema_version = int(
                connection.execute(
                    """SELECT value FROM meta
                       WHERE key = 'schema_version'"""
                ).fetchone()[0]
            )
        self.assertEqual(schema_version, SCHEMA_VERSION)

        read_only = Database(self.path, read_only=True)
        read_only.validate_current_schema()
        with read_only.read() as connection:
            question_info = {
                row["name"]: row
                for row in connection.execute(
                    "PRAGMA table_info(questions)"
                )
            }
            family_info = {
                row["name"]: row
                for row in connection.execute(
                    "PRAGMA table_info(learner_skill_families)"
                )
            }
            migrated_question = connection.execute(
                """SELECT content_hash FROM questions WHERE id = ?""",
                (self.question["id"],),
            ).fetchone()
            migrated_family = connection.execute(
                """SELECT kind, first_unguided_correct_at,
                          last_unguided_correct_at,
                          delayed_unguided_correct_at
                   FROM learner_skill_families
                   WHERE learner_id = ? AND concept_id = ?
                     AND family_id = ?""",
                (
                    "legacy-learner",
                    self.question["concept_id"],
                    self.question["family_id"],
                ),
            ).fetchone()
            violations = connection.execute(
                "PRAGMA foreign_key_check"
            ).fetchall()

        self.assertEqual(
            migrated_question["content_hash"],
            self.question["content_hash"],
        )
        self.assertEqual(migrated_family["kind"], self.question["kind"])
        self.assertEqual(
            migrated_family["first_unguided_correct_at"],
            "2020-01-01T00:00:00+00:00",
        )
        self.assertEqual(
            migrated_family["last_unguided_correct_at"],
            "2020-01-02T00:00:00+00:00",
        )
        self.assertEqual(
            migrated_family["delayed_unguided_correct_at"],
            "2020-01-03T00:00:00+00:00",
        )
        self.assertEqual(question_info["content_hash"]["notnull"], 1)
        self.assertEqual(family_info["kind"]["notnull"], 1)
        self.assertEqual(violations, [])

        # Re-running the writable initializer is semantically idempotent and
        # leaves the strict inspection contract green.
        self.database.initialize()
        Database(self.path, read_only=True).validate_current_schema()
        with self.database.read() as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) AS n FROM learner_skill_families"
                ).fetchone()["n"],
                1,
            )

    def test_migration_refuses_to_discard_an_unknown_question_column(
        self,
    ) -> None:
        self.make_nullable_v12_fixture()
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                "ALTER TABLE questions ADD COLUMN local_extension TEXT"
            )
            connection.commit()

        with self.assertRaisesRegex(
            ConflictError,
            "unexpected local_extension",
        ):
            self.database.initialize()

        with closing(sqlite3.connect(self.path)) as connection:
            columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(questions)"
                )
            }
            version = connection.execute(
                """SELECT value FROM meta
                   WHERE key = 'schema_version'"""
            ).fetchone()[0]
        self.assertIn("local_extension", columns)
        self.assertEqual(version, "12")


if __name__ == "__main__":
    unittest.main()
