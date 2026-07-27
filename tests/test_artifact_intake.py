# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import hashlib
import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tsq.artifact_intake import (
    CheckpointFileKind,
    MAX_PRODUCTIVE_ARTIFACT_BYTES,
    PreparedFileCheckpoint,
    ProductiveArtifactSnapshot,
    capture_productive_artifact,
    hash_productive_artifact,
    prepare_file_checkpoint,
)
from tsq.cli import main
from tsq.errors import ValidationError
from tsq.evidence import ActionKind


class ProductiveArtifactIntakeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_streaming_hash_and_closed_action_mappings_are_exact(self) -> None:
        material = (b"diagnostic trace\n" * 100_000) + b"final invariant\n"
        artifact = self.root / "private-debugging-notes.txt"
        artifact.write_bytes(material)
        expected = hashlib.sha256(material).hexdigest()

        self.assertEqual(hash_productive_artifact(artifact), expected)
        snapshot = capture_productive_artifact(artifact)
        self.assertIsInstance(snapshot, ProductiveArtifactSnapshot)
        self.assertEqual(snapshot.content, material)
        self.assertEqual(snapshot.sha256, expected)
        self.assertEqual(snapshot.size_bytes, len(material))
        self.assertNotIn("diagnostic trace", repr(snapshot))
        self.assertNotIn(artifact.name, repr(snapshot))
        prepared = {
            kind: prepare_file_checkpoint(
                artifact,
                kind=kind,
                artifact_kind=(
                    "diagnostic_trace_v1" if kind == "artifact" else None
                ),
            )
            for kind in ("artifact", "explanation", "submission")
        }
        self.assertEqual(
            prepared["artifact"].action_kind,
            ActionKind.ARTIFACT_CHECKPOINT,
        )
        self.assertEqual(
            dict(prepared["artifact"].payload),
            {
                "artifact_digest": expected,
                "artifact_kind": "diagnostic_trace_v1",
            },
        )
        self.assertEqual(
            prepared["explanation"].action_kind,
            ActionKind.EXPLANATION_CHECKPOINT,
        )
        self.assertEqual(
            dict(prepared["explanation"].payload),
            {"explanation_digest": expected},
        )
        self.assertEqual(
            prepared["submission"].action_kind,
            ActionKind.SUBMITTED,
        )
        self.assertEqual(
            dict(prepared["submission"].payload),
            {"submission_digest": expected},
        )

    def test_checkpoint_kind_and_artifact_identifier_fail_closed(self) -> None:
        artifact = self.root / "trace.json"
        artifact.write_text("{}", encoding="utf-8")
        cases = (
            (
                {"kind": "artifact"},
                "require --artifact-kind",
            ),
            (
                {"kind": "artifact", "artifact_kind": "../unsafe"},
                "stable identifier",
            ),
            (
                {"kind": "explanation", "artifact_kind": "not_allowed"},
                "only for artifact",
            ),
            (
                {"kind": "check_run"},
                "artifact, explanation, or submission",
            ),
        )
        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValidationError, message):
                    prepare_file_checkpoint(artifact, **arguments)

        with self.assertRaisesRegex(ValidationError, "does not match"):
            PreparedFileCheckpoint(
                kind=CheckpointFileKind.ARTIFACT,
                action_kind=ActionKind.CHECK_RUN,
                sha256="0" * 64,
                artifact_kind="diagnostic_trace_v1",
            )

    def test_empty_oversized_and_nonregular_files_are_rejected(self) -> None:
        empty = self.root / "empty.bin"
        empty.touch()
        maximum = self.root / "maximum.bin"
        with maximum.open("wb") as stream:
            stream.truncate(MAX_PRODUCTIVE_ARTIFACT_BYTES)
        for reader in (hash_productive_artifact, capture_productive_artifact):
            with self.subTest(reader=reader.__name__, case="maximum"):
                result = reader(maximum)
                digest = result if isinstance(result, str) else result.sha256
                self.assertRegex(digest, r"^[0-9a-f]{64}$")
        oversized = self.root / "oversized.bin"
        with oversized.open("wb") as stream:
            stream.truncate(MAX_PRODUCTIVE_ARTIFACT_BYTES + 1)
        directory = self.root / "directory"
        directory.mkdir()
        unsafe = (
            (empty, "must not be empty"),
            (oversized, "16 MiB"),
            (directory, "regular file"),
        )
        if hasattr(os, "mkfifo"):
            fifo = self.root / "artifact.fifo"
            os.mkfifo(fifo)
            unsafe += ((fifo, "regular file"),)
        for reader in (hash_productive_artifact, capture_productive_artifact):
            for path, message in unsafe:
                with self.subTest(reader=reader.__name__, path=path.name):
                    with self.assertRaisesRegex(
                        ValidationError, message
                    ) as raised:
                        reader(path)
                    self.assertNotIn(path.name, str(raised.exception))

    def test_symlink_and_missing_path_are_rejected_without_path_disclosure(
        self,
    ) -> None:
        target = self.root / "sensitive-target.txt"
        target.write_text("do not follow", encoding="utf-8")
        link = self.root / "learner-artifact-link"
        try:
            link.symlink_to(target)
        except (NotImplementedError, OSError):
            self.skipTest("symlinks are unavailable")
        missing = self.root / "missing-private-name"
        for reader in (hash_productive_artifact, capture_productive_artifact):
            with self.subTest(reader=reader.__name__, case="symlink"):
                with self.assertRaisesRegex(
                    ValidationError, "symlinks"
                ) as raised:
                    reader(link)
                self.assertNotIn(link.name, str(raised.exception))
            with self.subTest(reader=reader.__name__, case="missing"):
                with self.assertRaisesRegex(
                    ValidationError, "could not be inspected"
                ) as missing_error:
                    reader(missing)
                self.assertNotIn(
                    missing.name,
                    str(missing_error.exception),
                )

    def test_path_replacement_between_lstat_and_open_fails_before_read(
        self,
    ) -> None:
        artifact = self.root / "replace-me.bin"
        artifact.write_bytes(b"first artifact")
        real_open = os.open
        replaced = False

        def replacing_open(path, flags):
            nonlocal replaced
            if not replaced:
                replaced = True
                artifact.unlink()
                artifact.write_bytes(b"other artifact")
            return real_open(path, flags)

        with patch("tsq.artifact_intake.os.open", side_effect=replacing_open):
            with self.assertRaisesRegex(
                ValidationError, "changed before"
            ):
                hash_productive_artifact(artifact)

    def test_mid_read_mutation_and_short_read_fail_closed(self) -> None:
        artifact = self.root / "mutable.bin"
        original = b"A" * 4096
        artifact.write_bytes(original)
        real_read = os.read
        for reader in (hash_productive_artifact, capture_productive_artifact):
            with self.subTest(reader=reader.__name__, case="mutation"):
                artifact.write_bytes(original)
                original_stat = artifact.stat()
                mutated = False

                def mutating_read(descriptor, count):
                    nonlocal mutated
                    chunk = real_read(descriptor, count)
                    if chunk and not mutated:
                        mutated = True
                        artifact.write_bytes(b"B" * len(original))
                        os.utime(
                            artifact,
                            ns=(
                                original_stat.st_atime_ns,
                                original_stat.st_mtime_ns + 1_000_000,
                            ),
                        )
                    return chunk

                with patch(
                    "tsq.artifact_intake.os.read",
                    side_effect=mutating_read,
                ):
                    with self.assertRaisesRegex(
                        ValidationError, "changed while"
                    ):
                        reader(artifact)

            with self.subTest(reader=reader.__name__, case="short-read"):
                artifact.write_bytes(original)
                with patch("tsq.artifact_intake.os.read", return_value=b""):
                    with self.assertRaisesRegex(
                        ValidationError, "changed while"
                    ):
                        reader(artifact)

    def test_fifo_replacement_and_close_errors_fail_closed(self) -> None:
        artifact = self.root / "race-target.bin"
        artifact.write_bytes(b"regular bytes")
        if hasattr(os, "mkfifo"):
            real_open = os.open
            replaced = False

            def replacing_open(path, flags):
                nonlocal replaced
                if not replaced:
                    replaced = True
                    artifact.unlink()
                    os.mkfifo(artifact)
                return real_open(path, flags)

            with patch(
                "tsq.artifact_intake.os.open", side_effect=replacing_open
            ):
                with self.assertRaisesRegex(
                    ValidationError, "regular file"
                ):
                    hash_productive_artifact(artifact)

        artifact.unlink(missing_ok=True)
        artifact.write_bytes(b"close sentinel")
        real_close = os.close

        def closing_then_failing(descriptor):
            real_close(descriptor)
            raise OSError("synthetic close failure")

        with patch(
            "tsq.artifact_intake.os.close",
            side_effect=closing_then_failing,
        ):
            with self.assertRaisesRegex(
                ValidationError, "could not be closed safely"
            ):
                hash_productive_artifact(artifact)

        def read_failure(descriptor, count):
            raise OSError("synthetic read failure")

        with patch(
            "tsq.artifact_intake.os.read", side_effect=read_failure
        ), patch(
            "tsq.artifact_intake.os.close",
            side_effect=closing_then_failing,
        ):
            with self.assertRaisesRegex(
                ValidationError, "could not be read safely"
            ):
                hash_productive_artifact(artifact)

    def test_invalid_file_fails_before_database_initialization(self) -> None:
        database = self.root / "must-not-be-created.db"
        missing = self.root / "missing-artifact"
        output = io.StringIO()
        error = io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            exit_code = main(
                [
                    "--db",
                    str(database),
                    "task",
                    "checkpoint-file",
                    "attempt_does_not_matter",
                    str(missing),
                    "--kind",
                    "submission",
                    "--json",
                ]
            )
        self.assertEqual(exit_code, 2)
        self.assertFalse(database.exists())
        self.assertNotIn(missing.name, output.getvalue())
        self.assertNotIn(missing.name, error.getvalue())
        self.assertNotIn("Traceback", error.getvalue())

    def test_embedded_nul_fails_closed_before_database_initialization(
        self,
    ) -> None:
        database = self.root / "nul-must-not-be-created.db"
        private_path = "private\x00artifact"
        output = io.StringIO()
        error = io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            exit_code = main(
                [
                    "--db",
                    str(database),
                    "task",
                    "checkpoint-file",
                    "attempt_does_not_matter",
                    private_path,
                    "--kind",
                    "submission",
                    "--json",
                ]
            )
        self.assertEqual(exit_code, 2)
        self.assertFalse(database.exists())
        self.assertNotIn("private", output.getvalue())
        self.assertNotIn("private", error.getvalue())
        self.assertNotIn("Traceback", error.getvalue())


if __name__ == "__main__":
    unittest.main()
