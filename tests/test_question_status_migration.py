# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import base64
import gzip
import hashlib
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from tsq.models import QuestionStatus
from tsq.store import (
    SCHEMA_VERSION,
    Database,
    _capture_current_schema_contract,
    _expected_v22_schema_contract,
)

ROOT = Path(__file__).resolve().parents[1]
V0_1_0_FIXTURE = (
    ROOT / "tests" / "fixtures" / "v0_1_0_tiny_schema.sql.gz.b64"
)
V0_1_0_SQL_SHA256 = (
    "1a52ad7cc1ff3f9d17743f455b18d83a019bc6adb0290f697209a29a4c8f280b"
)
QUESTION_ID = "q_target"
RELEASE_ID = "rel_24a44e70e5c509c38341e77d"


def _install_released_v0_1_0_fixture(path: Path) -> None:
    """Restore a tiny database emitted by the annotated v0.1.0 source.

    The checked-in payload is a gzip-compressed ``sqlite3 .dump`` made after
    v0.1.0 initialized schema v22 and imported a sealed one-question corpus.
    It is intentionally not built with the current DDL or migration helpers.
    """

    encoded = "".join(V0_1_0_FIXTURE.read_text(encoding="ascii").splitlines())
    sql = gzip.decompress(base64.b64decode(encoded, validate=True))
    if hashlib.sha256(sql).hexdigest() != V0_1_0_SQL_SHA256:
        raise AssertionError("The annotated-v0.1.0 SQL fixture digest changed.")
    with closing(sqlite3.connect(path)) as connection:
        with connection:
            connection.executescript(sql.decode("utf-8"))


class QuestionStatusMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "question-status.db"
        _install_released_v0_1_0_fixture(self.path)
        self.release_id = RELEASE_ID
        with closing(sqlite3.connect(self.path)) as connection:
            self.release_before = connection.execute(
                "SELECT * FROM corpus_releases WHERE id=?",
                (self.release_id,),
            ).fetchone()
            self.membership_before = connection.execute(
                """SELECT * FROM release_questions
                   WHERE release_id=? ORDER BY question_id""",
                (self.release_id,),
            ).fetchall()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def upgrade(self) -> Database:
        database = Database(self.path)
        database.initialize()
        return database

    def test_fixture_is_the_exact_released_v22_schema(self) -> None:
        with closing(sqlite3.connect(self.path)) as connection:
            connection.row_factory = sqlite3.Row
            version = connection.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()["value"]
            contract = _capture_current_schema_contract(connection)

        self.assertEqual(version, "22")
        self.assertEqual(contract, _expected_v22_schema_contract())

    def test_v22_global_quarantine_migrates_to_draft(self) -> None:
        database = self.upgrade()

        with database.read() as connection:
            version = connection.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()["value"]
            status = connection.execute(
                "SELECT status FROM questions WHERE id=?",
                (QUESTION_ID,),
            ).fetchone()["status"]

        self.assertEqual(SCHEMA_VERSION, 23)
        self.assertEqual(version, "23")
        self.assertEqual(status, "draft")
        self.assertEqual(
            database.get_question(QUESTION_ID).status,
            QuestionStatus.DRAFT,
        )

    def test_v22_release_membership_bytes_survive_and_decode_as_draft(
        self,
    ) -> None:
        database = self.upgrade()

        with database.read() as connection:
            membership = connection.execute(
                """SELECT status, evidence_weight
                   FROM release_questions
                   WHERE release_id=? AND question_id=?""",
                (self.release_id, QUESTION_ID),
            ).fetchone()
            released = database._questions_by_ids(
                connection,
                [QUESTION_ID],
                release_id=self.release_id,
            )

        self.assertEqual(membership["status"], "quarantined")
        self.assertEqual(membership["evidence_weight"], 0.0)
        self.assertEqual(len(released), 1)
        self.assertEqual(released[0].status, QuestionStatus.DRAFT)
        with closing(sqlite3.connect(self.path)) as connection:
            release_after = connection.execute(
                "SELECT * FROM corpus_releases WHERE id=?",
                (self.release_id,),
            ).fetchone()
            membership_after = connection.execute(
                """SELECT * FROM release_questions
                   WHERE release_id=? ORDER BY question_id""",
                (self.release_id,),
            ).fetchall()
        self.assertEqual(release_after, self.release_before)
        self.assertEqual(membership_after, self.membership_before)
        integrity = database.verify_integrity()
        self.assertTrue(integrity["ok"], integrity["errors"])

    def test_v23_registry_rejects_legacy_quarantine_status(self) -> None:
        database = self.upgrade()

        with self.assertRaisesRegex(
            sqlite3.IntegrityError,
            "invalid question lifecycle status",
        ):
            with database.transaction() as connection:
                connection.execute(
                    "UPDATE questions SET status='quarantined' WHERE id=?",
                    (QUESTION_ID,),
                )


if __name__ == "__main__":
    unittest.main()
