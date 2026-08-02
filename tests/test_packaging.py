# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from contextlib import redirect_stderr
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

from scripts import sync_bundled_corpus
from tsq import __version__
from tsq.corpus import (
    corpus_source_digest,
    load_bundle,
    parse_bundle,
    parse_catalog,
    read_and_parse,
)
from tsq.cli import (
    BUNDLED_RELEASE_MARKER,
    BUNDLED_RESOURCE_DIGEST_MARKER,
    LEGACY_BUNDLED_RELEASE_HASHES,
    _ensure_starter_corpus,
)
from tsq.engine import AdaptiveEngine
from tsq.errors import ConflictError
from tsq.store import Database


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CORPUS = ROOT / "corpus"


class PackagingTestCase(unittest.TestCase):
    def test_public_release_license_and_attribution_are_consistent(self) -> None:
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(metadata["project"]["license"], "MPL-2.0")
        self.assertEqual(metadata["project"]["version"], __version__)

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
        self.assertIn("include tsq", manifest)
        self.assertIn("recursive-include corpus *.json *.md", manifest)
        self.assertFalse(
            any(line.startswith("recursive-include docs ") for line in manifest)
        )
        for source_file in (ROOT / "src" / "tsq").rglob("*.py"):
            with self.subTest(source_file=source_file.relative_to(ROOT)):
                self.assertIn(
                    "# SPDX-License-Identifier: MPL-2.0",
                    source_file.read_text().splitlines()[:3],
                )

    def test_cli_reports_the_package_version(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "src")
        result = subprocess.run(
            [sys.executable, "-m", "tsq", "--version"],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, f"tsq {__version__}\n")

    def test_bundled_seed_tree_is_byte_identical_to_canonical_corpus(self) -> None:
        source = {
            path.relative_to(SOURCE_CORPUS).as_posix(): path.read_bytes()
            for path in SOURCE_CORPUS.rglob("*.json")
        }
        resource_root = files("tsq.data").joinpath("curriculum")

        def resource_inventory(resource, prefix: tuple[str, ...] = ()):
            inventory = {}
            for child in resource.iterdir():
                relative = (*prefix, child.name)
                if child.is_dir():
                    inventory.update(resource_inventory(child, relative))
                elif child.name.endswith(".json"):
                    inventory["/".join(relative)] = child.read_bytes()
            return inventory

        bundled = resource_inventory(resource_root)
        self.assertEqual(bundled, source)

    def test_sharding_preserves_the_semantic_release_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Database(Path(temporary) / "release.db")
            database.initialize()
            result = database.import_corpus(
                *read_and_parse(SOURCE_CORPUS, include_catalog=True)
            )

        self.assertEqual(
            result["release_id"],
            "rel_e8047a6bdb517fd42e1cb2d6",
        )
        self.assertEqual(result["questions"], 288)
        self.assertEqual(result["topics"], 16)

    def test_sync_refuses_invalid_source_before_touching_packaged_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            packaged = root / "packaged"
            shutil.copytree(SOURCE_CORPUS, source)
            shutil.copytree(
                ROOT / "src" / "tsq" / "data" / "curriculum",
                packaged,
            )
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema_version"] = 2
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            def snapshot(path: Path) -> dict[str, bytes]:
                return {
                    item.relative_to(path).as_posix(): item.read_bytes()
                    for item in path.rglob("*")
                    if item.is_file()
                }

            before = snapshot(packaged)
            with (
                patch.object(sync_bundled_corpus, "SOURCE", source),
                patch.object(sync_bundled_corpus, "PACKAGED", packaged),
                redirect_stderr(io.StringIO()),
            ):
                code = sync_bundled_corpus.main(["--write"])

            self.assertEqual(code, 1)
            self.assertEqual(snapshot(packaged), before)

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
            self.assertEqual(payload["corpus"], "tsq.data:curriculum")
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

    def test_current_starter_lineage_is_an_exact_noop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Database(Path(temporary) / "current.db")
            database.initialize()

            first = _ensure_starter_corpus(database)
            with database.read() as connection:
                before = "\n".join(connection.iterdump())
                active = connection.execute(
                    "SELECT value FROM meta WHERE key='active_corpus_release'"
                ).fetchone()["value"]
            repeated = _ensure_starter_corpus(database)
            with database.read() as connection:
                after = "\n".join(connection.iterdump())
                marker = connection.execute(
                    "SELECT value FROM meta WHERE key=?",
                    (BUNDLED_RELEASE_MARKER,),
                ).fetchone()["value"]

            self.assertTrue(first)
            self.assertFalse(repeated)
            self.assertIsNone(repeated.retained_release_id)
            self.assertIsNone(repeated.conflict)
            self.assertEqual(marker, active)
            self.assertEqual(after, before)
            self.assertTrue(database.verify_integrity()["ok"])

    def test_starter_migrates_the_old_monolith_digest_without_a_new_release(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Database(Path(temporary) / "old-resource-marker.db")
            database.initialize()
            imported = database.import_corpus(
                *read_and_parse(SOURCE_CORPUS, include_catalog=True)
            )
            release_id = str(imported["release_id"])
            with database.transaction() as connection:
                connection.executemany(
                    "INSERT INTO meta(key, value) VALUES (?, ?)",
                    (
                        (BUNDLED_RELEASE_MARKER, release_id),
                        (
                            BUNDLED_RESOURCE_DIGEST_MARKER,
                            "ad5d34693371d606e318cfd51cc61913894390d578b1810a45b9af43190302b0",
                        ),
                    ),
                )

            status = _ensure_starter_corpus(database)

            self.assertFalse(status)
            with database.read() as connection:
                active = connection.execute(
                    "SELECT value FROM meta WHERE key='active_corpus_release'"
                ).fetchone()["value"]
                marker = connection.execute(
                    "SELECT value FROM meta WHERE key=?",
                    (BUNDLED_RESOURCE_DIGEST_MARKER,),
                ).fetchone()["value"]
                release_count = connection.execute(
                    "SELECT COUNT(*) AS n FROM corpus_releases"
                ).fetchone()["n"]
            self.assertEqual(active, release_id)
            self.assertEqual(marker, corpus_source_digest(SOURCE_CORPUS))
            self.assertEqual(release_count, 1)

    def test_starter_tracks_its_lineage_without_overriding_custom_corpus(self) -> None:
        bundle = load_bundle(SOURCE_CORPUS)
        # Remove one explicitly marked generated/quarantined item so the
        # immutable legacy-unattested cohort remains complete and byte-stable.
        removed_id = "q_attention_runtime_workspace_boundary_001"
        bundle["questions"] = [
            question
            for question in bundle["questions"]
            if question["id"] != removed_id
        ]
        parsed = parse_bundle(bundle)
        domains, topics = parse_catalog(bundle, parsed[0], parsed[4])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "custom.db"
            database = Database(path)
            database.initialize()
            custom_release = database.import_corpus(
                *parsed, domains, topics
            )["release_id"]

            self.assertFalse(_ensure_starter_corpus(database))
            with database.read() as connection:
                active = connection.execute(
                    "SELECT value FROM meta WHERE key='active_corpus_release'"
                ).fetchone()["value"]
                marker = connection.execute(
                    "SELECT value FROM meta WHERE key = ?",
                    (BUNDLED_RELEASE_MARKER,),
                ).fetchone()
            self.assertEqual(active, custom_release)
            self.assertIsNone(marker)

            with database.transaction() as connection:
                connection.execute(
                    "INSERT INTO meta(key, value) VALUES (?, ?)",
                    (BUNDLED_RELEASE_MARKER, custom_release),
                )
            self.assertTrue(_ensure_starter_corpus(database))
            with database.read() as connection:
                upgraded = connection.execute(
                    "SELECT value FROM meta WHERE key='active_corpus_release'"
                ).fetchone()["value"]
                marker = connection.execute(
                    "SELECT value FROM meta WHERE key = ?",
                    (BUNDLED_RELEASE_MARKER,),
                ).fetchone()["value"]
            self.assertNotEqual(upgraded, custom_release)
            self.assertEqual(marker, upgraded)

    def test_starter_does_not_replace_uncataloged_custom_corpus(self) -> None:
        bundle = load_bundle(SOURCE_CORPUS)
        bundle.pop("domains")
        bundle.pop("topics")
        removed_id = "q_attention_runtime_workspace_boundary_001"
        bundle["questions"] = [
            question
            for question in bundle["questions"]
            if question["id"] != removed_id
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "uncataloged-custom.db"
            database = Database(path)
            database.initialize()
            custom_release = database.import_corpus(
                *parse_bundle(bundle)
            )["release_id"]

            status = _ensure_starter_corpus(database)

            self.assertFalse(status)
            self.assertIsNone(status.retained_release_id)
            self.assertIsNone(status.conflict)
            with database.read() as connection:
                active = connection.execute(
                    "SELECT value FROM meta WHERE key='active_corpus_release'"
                ).fetchone()["value"]
                marker = connection.execute(
                    "SELECT value FROM meta WHERE key = ?",
                    (BUNDLED_RELEASE_MARKER,),
                ).fetchone()
                release_count = connection.execute(
                    "SELECT COUNT(*) AS n FROM corpus_releases"
                ).fetchone()["n"]
            self.assertEqual(active, custom_release)
            self.assertIsNone(marker)
            self.assertEqual(release_count, 1)
            self.assertTrue(database.verify_integrity()["ok"])

    def test_untrusted_registry_conflict_is_not_silently_retained(self) -> None:
        bundle = load_bundle(SOURCE_CORPUS)
        source = next(
            item
            for item in bundle["sources"]
            if item["id"] == "src_vaswani_attention_2017"
        )
        source["uri"] = "https://arxiv.org/abs/1706.03762"
        parsed = parse_bundle(bundle)
        domains, topics = parse_catalog(bundle, parsed[0], parsed[4])
        with tempfile.TemporaryDirectory() as temporary:
            database = Database(Path(temporary) / "untrusted-conflict.db")
            database.initialize()
            custom_release = database.import_corpus(
                *parsed, domains, topics
            )["release_id"]
            with database.transaction() as connection:
                connection.execute(
                    "DELETE FROM meta WHERE key='active_corpus_release'"
                )
            with database.read() as connection:
                before = "\n".join(connection.iterdump())

            with self.assertRaisesRegex(
                ConflictError,
                "Source src_vaswani_attention_2017 is immutable",
            ):
                _ensure_starter_corpus(database)

            with database.read() as connection:
                after = "\n".join(connection.iterdump())
                release_ids = [
                    row["id"]
                    for row in connection.execute(
                        "SELECT id FROM corpus_releases ORDER BY id"
                    )
                ]
                marker = connection.execute(
                    "SELECT value FROM meta WHERE key=?",
                    (BUNDLED_RELEASE_MARKER,),
                ).fetchone()
            self.assertEqual(after, before)
            self.assertEqual(release_ids, [custom_release])
            self.assertIsNone(marker)

    def test_answer_cli_rejects_oversize_integer_without_traceback_or_write(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            isolated = Path(temporary)
            database_path = isolated / "bounded-answer.db"
            database = Database(database_path)
            database.initialize()
            database.import_corpus(
                *parse_bundle(load_bundle(SOURCE_CORPUS))
            )
            engine = AdaptiveEngine(database)
            engine.create_learner("bounded-cli")
            session = engine.start_session(
                "bounded-cli", "c_transformers", seed=17
            )
            presentation = engine.next_question(session["id"])

            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(ROOT / "src")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tsq",
                    "--db",
                    str(database_path),
                    "answer",
                    presentation.decision_id,
                    presentation.question.correct_option.id,
                    "--response-ms",
                    str(2**100),
                ],
                cwd=isolated,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("response_ms must be an integer", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            with database.read() as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) AS n FROM attempts"
                    ).fetchone()["n"],
                    0,
                )

    def test_start_upgrades_marked_legacy_catalog_without_rewriting_history(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            isolated = Path(temporary)
            database_path = isolated / "legacy.db"
            bundle = load_bundle(SOURCE_CORPUS)
            bundle.pop("domains")
            bundle.pop("topics")
            legacy_source = next(
                source
                for source in bundle["sources"]
                if source["id"] == "src_vaswani_attention_2017"
            )
            legacy_source.pop("uri", None)

            database = Database(database_path)
            database.initialize()
            legacy_release = database.import_corpus(*parse_bundle(bundle))["release_id"]
            with database.transaction() as connection:
                connection.execute(
                    "INSERT INTO meta(key, value) VALUES (?, ?)",
                    (BUNDLED_RELEASE_MARKER, legacy_release),
                )
            engine = AdaptiveEngine(database)
            engine.create_learner("legacy-learner", "Legacy Learner")
            legacy_session = engine.start_session(
                "legacy-learner", "c_transformers", seed=11
            )

            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(ROOT / "src")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tsq",
                    "--db",
                    str(database_path),
                    "start",
                    "--learner",
                    "upgraded-learner",
                    "--topic",
                    "Large Language Models",
                    "--seed",
                    "13",
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
            self.assertIn(
                "Installed the bundled reviewed curriculum catalog.", result.stdout
            )
            with database.read() as connection:
                active_release = connection.execute(
                    "SELECT value FROM meta WHERE key = 'active_corpus_release'"
                ).fetchone()["value"]
                pinned_release = connection.execute(
                    "SELECT corpus_release_id FROM sessions WHERE id = ?",
                    (legacy_session["id"],),
                ).fetchone()["corpus_release_id"]
                release_count = connection.execute(
                    "SELECT COUNT(*) AS n FROM corpus_releases"
                ).fetchone()["n"]
                legacy_topics = connection.execute(
                    "SELECT COUNT(*) AS n FROM release_topics WHERE release_id = ?",
                    (legacy_release,),
                ).fetchone()["n"]
                active_topics = connection.execute(
                    "SELECT COUNT(*) AS n FROM release_topics WHERE release_id = ?",
                    (active_release,),
                ).fetchone()["n"]

            self.assertNotEqual(active_release, legacy_release)
            self.assertEqual(pinned_release, legacy_release)
            self.assertEqual(release_count, 2)
            self.assertEqual(legacy_topics, 0)
            self.assertGreater(active_topics, 0)
            self.assertTrue(database.verify_integrity()["ok"])

    def test_start_retains_valid_release_when_bundled_ids_conflict(self) -> None:
        """A historical metadata collision cannot make an install unusable."""

        with tempfile.TemporaryDirectory() as temporary:
            isolated = Path(temporary)
            database_path = isolated / "immutable-source.db"
            bundle = load_bundle(SOURCE_CORPUS)
            historical_source = next(
                source
                for source in bundle["sources"]
                if source["id"] == "src_vaswani_attention_2017"
            )
            historical_source["uri"] = "https://arxiv.org/abs/1706.03762"
            parsed = parse_bundle(bundle)
            domains, topics = parse_catalog(bundle, parsed[0], parsed[4])

            database = Database(database_path)
            database.initialize()
            retained_release = database.import_corpus(
                *parsed, domains, topics
            )["release_id"]
            with database.transaction() as connection:
                connection.execute(
                    "INSERT INTO meta(key, value) VALUES (?, ?)",
                    (BUNDLED_RELEASE_MARKER, retained_release),
                )

            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(ROOT / "src")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tsq",
                    "--db",
                    str(database_path),
                    "start",
                    "--learner",
                    "immutable-source-user",
                    "--topic",
                    "Large Language Models",
                    "--seed",
                    "17",
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
            self.assertIn("Large Language Models", result.stdout)
            self.assertIn("Session stopped", result.stdout)
            self.assertIn(
                "Bundled curriculum update was withheld to preserve immutable "
                f"release {retained_release}",
                result.stderr,
            )
            self.assertIn("Continuing with that sealed release", result.stderr)
            self.assertIn("TSQ_DB=tsq-latest.db ./start", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            with database.read() as connection:
                active_release = connection.execute(
                    "SELECT value FROM meta WHERE key = 'active_corpus_release'"
                ).fetchone()["value"]
                release_count = connection.execute(
                    "SELECT COUNT(*) AS n FROM corpus_releases"
                ).fetchone()["n"]
            self.assertEqual(active_release, retained_release)
            self.assertEqual(release_count, 1)
            self.assertTrue(database.verify_integrity()["ok"])

            with database.read() as connection:
                bundle_hash = connection.execute(
                    "SELECT bundle_hash FROM corpus_releases WHERE id=?",
                    (retained_release,),
                ).fetchone()["bundle_hash"]
            self.assertNotIn(bundle_hash, LEGACY_BUNDLED_RELEASE_HASHES)

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

    def test_repository_cli_launcher_exposes_full_command_surface(self) -> None:
        launcher = ROOT / "tsq"
        self.assertTrue(launcher.is_file())
        self.assertTrue(os.access(launcher, os.X_OK))
        self.assertIn("SPDX-License-Identifier: MPL-2.0", launcher.read_text())

        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "launcher.db"
            result = subprocess.run(
                [
                    str(launcher),
                    "--db",
                    str(database),
                    "init",
                    "--json",
                ],
                cwd=temporary,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertGreaterEqual(payload["questions"], 20)
            self.assertTrue(database.is_file())


if __name__ == "__main__":
    unittest.main()
