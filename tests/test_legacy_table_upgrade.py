# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

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

from tests.schema_upgrade_helpers import (
    durable_database_fingerprint,
    restore_pre_shadow_schema,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"


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
            restore_pre_shadow_schema(connection)
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

    @staticmethod
    def make_noncanonical_question_fixture(
        database: Database,
        *,
        schema_version: int,
    ) -> tuple[tuple[str, str], ...]:
        """Reconstruct the nullable/trailing column emitted by legacy ALTER."""

        with database.transaction() as connection:
            restore_pre_shadow_schema(connection)
            hashes = tuple(
                (row["id"], row["content_hash"])
                for row in connection.execute(
                    "SELECT id, content_hash FROM questions ORDER BY id"
                ).fetchall()
            )
            database._drop_corpus_registry_triggers(connection)
            connection.execute(
                "ALTER TABLE questions DROP COLUMN content_hash"
            )
            connection.execute(
                "ALTER TABLE questions ADD COLUMN content_hash TEXT"
            )
            connection.executemany(
                """UPDATE questions SET content_hash=?
                   WHERE id=?""",
                (
                    (content_hash, question_id)
                    for question_id, content_hash in hashes
                ),
            )
            connection.execute(
                """UPDATE meta SET value=?
                   WHERE key='schema_version'""",
                (str(schema_version),),
            )
        return hashes

    def rebuild_learners_with_unknown_check(self) -> str:
        """Give a legacy-labelled table an unrecognized same-column shape."""

        with closing(sqlite3.connect(self.path)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=OFF")
            connection.execute("BEGIN IMMEDIATE")
            restore_pre_shadow_schema(connection)
            row = connection.execute(
                """SELECT sql FROM sqlite_master
                   WHERE type='table' AND name='learners'"""
            ).fetchone()
            if row is None or not row["sql"]:
                raise AssertionError("Fixture lacks the learners table.")
            canonical_sql = row["sql"]
            closing_parenthesis = canonical_sql.rfind(")")
            if closing_parenthesis < 0:
                raise AssertionError(
                    "Learners table has no closing definition boundary."
                )
            altered_sql = (
                canonical_sql[:closing_parenthesis]
                + ", CHECK(length(display_name) < 10000)"
                + canonical_sql[closing_parenthesis:]
            ).replace(
                "CREATE TABLE learners",
                "CREATE TABLE _tampered_learners",
                1,
            )
            if altered_sql == canonical_sql:
                raise AssertionError(
                    "Fixture did not alter the learners definition."
                )
            triggers = tuple(
                (row["name"], row["sql"])
                for row in connection.execute(
                    """SELECT name, sql FROM sqlite_master
                       WHERE type='trigger' AND sql IS NOT NULL
                       ORDER BY name"""
                )
            )
            for name, _sql in triggers:
                quoted_name = '"' + name.replace('"', '""') + '"'
                connection.execute(f"DROP TRIGGER {quoted_name}")
            connection.execute(altered_sql)
            connection.execute(
                """INSERT INTO _tampered_learners(
                       id, display_name, revision, created_at
                   )
                   SELECT id, display_name, revision, created_at
                   FROM learners"""
            )
            connection.execute("DROP TABLE learners")
            connection.execute(
                """ALTER TABLE _tampered_learners
                   RENAME TO learners"""
            )
            for _name, sql in triggers:
                connection.execute(sql)
            connection.execute(
                """UPDATE meta SET value='12'
                   WHERE key='schema_version'"""
            )
            connection.commit()
            stored = connection.execute(
                """SELECT sql FROM sqlite_master
                   WHERE type='table' AND name='learners'"""
            ).fetchone()
        if stored is None or not stored["sql"]:
            raise AssertionError("Tampered learners definition was not stored.")
        return stored["sql"]

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
        self.assertIn(
            f"current schema {SCHEMA_VERSION}",
            error.getvalue(),
        )
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

    def test_noncanonical_v13_v14_v15_sources_are_canonicalized(
        self,
    ) -> None:
        for source_version in (13, 14, 15):
            with self.subTest(source_version=source_version):
                with tempfile.TemporaryDirectory() as directory:
                    path = (
                        Path(directory)
                        / f"legacy-v{source_version}-noncanonical.db"
                    )
                    database = Database(path)
                    database.initialize()
                    database.import_corpus(
                        *read_and_parse(CORPUS, include_catalog=True)
                    )
                    before_hashes = (
                        self.make_noncanonical_question_fixture(
                            database,
                            schema_version=source_version,
                        )
                    )
                    with database.read() as connection:
                        before_rows = tuple(
                            (row["id"], row["content_hash"])
                            for row in connection.execute(
                                """SELECT id, content_hash
                                   FROM questions ORDER BY id"""
                            )
                        )
                        before_info = {
                            row["name"]: row
                            for row in connection.execute(
                                "PRAGMA table_info(questions)"
                            )
                        }
                    self.assertEqual(before_rows, before_hashes)
                    self.assertEqual(
                        before_info["content_hash"]["notnull"],
                        0,
                    )

                    database.initialize()

                    Database(
                        path,
                        read_only=True,
                    ).validate_current_schema()
                    with database.read() as connection:
                        after_rows = tuple(
                            (row["id"], row["content_hash"])
                            for row in connection.execute(
                                """SELECT id, content_hash
                                   FROM questions ORDER BY id"""
                            )
                        )
                        after_info = {
                            row["name"]: row
                            for row in connection.execute(
                                "PRAGMA table_info(questions)"
                            )
                        }
                        version = connection.execute(
                            """SELECT value FROM meta
                               WHERE key='schema_version'"""
                        ).fetchone()["value"]
                        violations = connection.execute(
                            "PRAGMA foreign_key_check"
                        ).fetchall()
                    self.assertEqual(after_rows, before_hashes)
                    self.assertEqual(
                        after_info["content_hash"]["notnull"],
                        1,
                    )
                    self.assertEqual(version, str(SCHEMA_VERSION))
                    self.assertEqual(violations, [])

    def test_rebuild_rejects_unknown_index_and_trigger_without_mutation(
        self,
    ) -> None:
        self.make_nullable_v12_fixture()
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                """CREATE INDEX local_questions_family
                   ON questions(family_id, id)"""
            )
            connection.execute(
                """CREATE TRIGGER local_questions_probe
                   AFTER INSERT ON questions BEGIN
                       SELECT 1;
                   END"""
            )
            connection.commit()
            expected_objects = tuple(
                connection.execute(
                    """SELECT type, name, sql FROM sqlite_master
                       WHERE name IN (
                           'local_questions_family',
                           'local_questions_probe'
                       )
                       ORDER BY type, name"""
                )
            )
        before_files = durable_database_fingerprint(self.path)

        with self.assertRaises(ConflictError):
            self.database.initialize()

        self.assertEqual(durable_database_fingerprint(self.path), before_files)
        with closing(sqlite3.connect(self.path)) as connection:
            actual_objects = tuple(
                connection.execute(
                    """SELECT type, name, sql FROM sqlite_master
                       WHERE name IN (
                           'local_questions_family',
                           'local_questions_probe'
                       )
                       ORDER BY type, name"""
                )
            )
            version = connection.execute(
                """SELECT value FROM meta
                   WHERE key='schema_version'"""
            ).fetchone()[0]
            question_count = connection.execute(
                "SELECT COUNT(*) FROM questions"
            ).fetchone()[0]
        self.assertEqual(actual_objects, expected_objects)
        self.assertEqual(version, "12")
        self.assertGreater(question_count, 0)

    def test_rebuild_rejects_unknown_same_column_definition_atomically(
        self,
    ) -> None:
        altered_sql = self.rebuild_learners_with_unknown_check()
        before_files = durable_database_fingerprint(self.path)
        with closing(sqlite3.connect(self.path)) as connection:
            before_learners = tuple(
                connection.execute(
                    "SELECT * FROM learners ORDER BY id"
                )
            )

        with self.assertRaises(ConflictError):
            self.database.initialize()

        self.assertEqual(durable_database_fingerprint(self.path), before_files)
        with closing(sqlite3.connect(self.path)) as connection:
            stored = connection.execute(
                """SELECT sql FROM sqlite_master
                   WHERE type='table' AND name='learners'"""
            ).fetchone()[0]
            after_learners = tuple(
                connection.execute(
                    "SELECT * FROM learners ORDER BY id"
                )
            )
            version = connection.execute(
                """SELECT value FROM meta
                   WHERE key='schema_version'"""
            ).fetchone()[0]
        self.assertEqual(stored, altered_sql)
        self.assertEqual(after_learners, before_learners)
        self.assertEqual(version, "12")


if __name__ == "__main__":
    unittest.main()
