# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from importlib.resources import files
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CORPUS = ROOT / "corpus" / "ai_curriculum.json"


class PackagingTestCase(unittest.TestCase):
    def test_public_release_license_and_attribution_are_consistent(self) -> None:
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(metadata["project"]["license"], "MPL-2.0")

        license_text = (ROOT / "LICENSE").read_text()
        self.assertTrue(license_text.startswith("Mozilla Public License Version 2.0\n"))
        self.assertIn("Exhibit A - Source Code Form License Notice", license_text)
        self.assertIn("Exhibit B - \"Incompatible With Secondary Licenses\" Notice", license_text)

        notice = (ROOT / "NOTICE").read_text()
        self.assertIn("Copyright 2026 The Second Question contributors", notice)
        self.assertIn("Mozilla Public License", notice)

        manifest = (ROOT / "MANIFEST.in").read_text().splitlines()
        self.assertIn("include LICENSE", manifest)
        self.assertIn("include NOTICE", manifest)
        self.assertIn("include start", manifest)
        self.assertFalse(
            any(line.startswith("recursive-include docs ") for line in manifest)
        )
        for source_file in (ROOT / "src" / "tsq").rglob("*.py"):
            with self.subTest(source_file=source_file.relative_to(ROOT)):
                self.assertIn(
                    "# SPDX-License-Identifier: MPL-2.0",
                    source_file.read_text().splitlines()[:3],
                )

    def test_bundled_seed_is_byte_identical_to_canonical_corpus(self) -> None:
        source = SOURCE_CORPUS.read_bytes()
        bundled = files("tsq.data").joinpath("ai_curriculum.json").read_bytes()

        self.assertEqual(hashlib.sha256(bundled).digest(), hashlib.sha256(source).digest())
        self.assertEqual(bundled, source)

    def test_init_default_does_not_depend_on_checkout_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            isolated = Path(temporary)
            database = isolated / "installed-default.db"
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(ROOT / "src")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tsq",
                    "--db",
                    str(database),
                    "init",
                    "--json",
                ],
                cwd=isolated,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["corpus"], "tsq.data:ai_curriculum.json")
            self.assertGreaterEqual(payload["questions"], 20)
            self.assertTrue(database.is_file())

    def test_start_command_bootstraps_catalog_and_uses_friendly_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            isolated = Path(temporary)
            database = isolated / "starter.db"
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(ROOT / "src")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tsq",
                    "--db",
                    str(database),
                    "start",
                    "--learner",
                    "starter",
                    "--seed",
                    "7",
                ],
                cwd=isolated,
                env=environment,
                input="q\n",
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Installed the bundled reviewed curriculum catalog.", result.stdout)
            self.assertIn("Large Language Models", result.stdout)
            self.assertIn("Session stopped", result.stdout)
            self.assertTrue(database.is_file())

    def test_repository_start_launcher_is_executable(self) -> None:
        launcher = ROOT / "start"
        self.assertTrue(launcher.is_file())
        self.assertTrue(os.access(launcher, os.X_OK))
        self.assertIn("SPDX-License-Identifier: MPL-2.0", launcher.read_text())

        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "launcher.db"
            environment = os.environ.copy()
            environment["TSQ_DB"] = str(database)
            result = subprocess.run(
                [
                    str(launcher),
                    "--learner",
                    "launcher-test",
                    "--topic",
                    "Transformers",
                    "--seed",
                    "5",
                ],
                cwd=temporary,
                env=environment,
                input="q\n",
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Transformers", result.stdout)
            self.assertIn("Session stopped", result.stdout)
            self.assertTrue(database.is_file())


if __name__ == "__main__":
    unittest.main()
