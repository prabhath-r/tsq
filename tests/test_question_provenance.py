# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from tests.test_store_integrity import tiny_corpus
from tsq.corpus import load_bundle, read_and_parse, validate_bundle
from tsq.errors import ValidationError
from tsq.models import QuestionStatus
from tsq.provenance import (
    LEGACY_UNATTESTED_COHORT_SHA256,
    legacy_unattested_cohort_digest,
)
from tsq.store import Database, _legacy_question_identity


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"


class RawQuestionProvenanceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = load_bundle(CORPUS)
        self.question = next(
            question
            for question in self.bundle["questions"]
            if question["status"] == "approved"
            and "generated" not in question.get("provenance", {})
        )

    def provenance_codes(self) -> set[str]:
        return {
            issue.code
            for issue in validate_bundle(self.bundle)
            if issue.question_id == self.question["id"]
            and (
                "provenance" in issue.code
                or issue.code.endswith("_flag_type")
            )
        }

    def test_public_question_provenance_rejects_identity_aliases_recursively(
        self,
    ) -> None:
        for identity_claim in (
            {"provider": "private-operational-identity"},
            {"model": "private-operational-identity"},
            {"generator": "private-operational-identity"},
            {"provider_name": "private-operational-identity"},
            {"model_name": "private-operational-identity"},
            {"metadata": {"provider": "private-operational-identity"}},
            {"metadata": [{"modelName": "private-operational-identity"}]},
            {"generator_identity": "private-operational-identity"},
            {"vendor_id": "private-operational-identity"},
            {"engineName": "private-operational-identity"},
            {"modelidentity": "private-operational-identity"},
            {"llm_name": "private-operational-identity"},
            {"backendName": "private-operational-identity"},
            {"model_reviewer": "private-operational-identity"},
            {"generator_reviewer": "private-operational-identity"},
            {"llm_backend": "private-operational-identity"},
            {"engine_label": "private-operational-identity"},
            {"authoring_backend_label": "private-operational-identity"},
        ):
            with self.subTest(identity_claim=identity_claim):
                self.question["status"] = "draft"
                self.question["provenance"] = {
                    "generated": True,
                    "human_review": False,
                    **identity_claim,
                }
                self.assertIn(
                    "public_provenance_identity_forbidden",
                    self.provenance_codes(),
                )

    def test_public_question_provenance_allows_nonidentity_commitments(
        self,
    ) -> None:
        self.question["status"] = "draft"
        self.question["provenance"] = {
            "generated": True,
            "human_review": False,
            "generator_output_sha256": "a" * 64,
            "generator_provenance_sha256": "b" * 64,
            "generator_declared_provenance_sha256": "c" * 64,
            "independent_model_review_count": 2,
        }

        self.assertEqual(self.provenance_codes(), set())

    def test_exact_packaged_legacy_cohort_retains_compatibility(self) -> None:
        legacy = [
            question
            for question in self.bundle["questions"]
            if "generated" not in question.get("provenance", {})
        ]

        self.assertEqual(len(legacy), 222)
        self.assertFalse(
            any(
                issue.code == "generated_provenance_required"
                for issue in validate_bundle(self.bundle)
            )
        )

    def test_exact_legacy_member_can_be_omitted_from_a_later_bundle(self) -> None:
        revision_parents = {
            question.get("revision_of")
            for question in self.bundle["questions"]
            if question.get("revision_of")
        }
        omitted_id = next(
            question["id"]
            for question in self.bundle["questions"]
            if "generated" not in question.get("provenance", {})
            and question["id"] not in revision_parents
        )
        self.bundle["questions"] = [
            question
            for question in self.bundle["questions"]
            if question["id"] != omitted_id
        ]

        self.assertFalse(
            any(
                issue.code == "generated_provenance_required"
                for issue in validate_bundle(self.bundle)
            )
        )

    def test_raw_and_typed_paths_agree_on_packaged_cohort(self) -> None:
        parsed = read_and_parse(CORPUS, include_catalog=True)
        typed_legacy = [
            question
            for question in parsed[4]
            if "generated" not in question.provenance
        ]

        self.assertFalse(validate_bundle(self.bundle))
        self.assertEqual(
            legacy_unattested_cohort_digest(
                _legacy_question_identity(question)
                for question in typed_legacy
            ),
            LEGACY_UNATTESTED_COHORT_SHA256,
        )

    def test_legacy_content_change_invalidates_manifest_commitment(self) -> None:
        self.question["stem"] += " Unattested content mutation."

        self.assertIn(
            "generated_provenance_required",
            self.provenance_codes(),
        )

    def test_legacy_objective_binding_change_invalidates_manifest(self) -> None:
        self.question = next(
            question
            for question in self.bundle["questions"]
            if "generated" not in question.get("provenance", {})
            and question.get("learning_objective_id")
        )
        self.question["learning_objective_id"] += "_mutated"

        self.assertIn(
            "generated_provenance_required",
            self.provenance_codes(),
        )

    def test_legacy_diagnostic_binding_change_invalidates_manifest(self) -> None:
        self.question = next(
            question
            for question in self.bundle["questions"]
            if "generated" not in question.get("provenance", {})
            and any(
                "diagnostic_objective_id" in option
                for option in question["options"]
            )
        )
        option = next(
            option
            for option in self.question["options"]
            if "diagnostic_objective_id" in option
        )
        option["diagnostic_objective_id"] += "_mutated"

        self.assertIn(
            "generated_provenance_required",
            self.provenance_codes(),
        )

    def test_legacy_lifecycle_change_does_not_change_content_identity(
        self,
    ) -> None:
        self.question["status"] = "retired"

        self.assertNotIn(
            "generated_provenance_required",
            self.provenance_codes(),
        )

    def test_new_revision_must_declare_generation_provenance(self) -> None:
        revision = deepcopy(self.question)
        revision.update(
            {
                "id": self.question["id"] + "_revision",
                "version": self.question["version"] + 1,
                "status": "draft",
                "stem": self.question["stem"] + " New revision.",
                "revision_of": self.question["id"],
            }
        )
        self.bundle["questions"].append(revision)
        self.question = revision

        self.assertIn(
            "generated_provenance_required",
            self.provenance_codes(),
        )

        revision["provenance"] = {
            **revision["provenance"],
            "generated": False,
        }
        self.assertFalse(
            any(
                issue.code == "generated_provenance_required"
                for issue in validate_bundle(self.bundle)
            )
        )

    def test_generated_and_human_review_flags_are_exact_booleans(self) -> None:
        self.question["provenance"] = {
            "generated": "true",
            "human_review": "true",
        }

        self.assertEqual(
            self.provenance_codes(),
            {"generated_flag_type", "human_review_flag_type"},
        )


class TypedQuestionProvenanceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.tempdir.name) / "provenance.db")
        self.database.initialize()
        self.bundle = tiny_corpus()
        self.expert = self.bundle[-1][0]

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_typed_fixture_requires_explicit_generation_provenance(self) -> None:
        missing_marker = replace(
            self.expert,
            provenance={"authoring_method": "uncommitted-fixture"},
        )

        with self.assertRaisesRegex(
            ValidationError,
            "generated_provenance_required",
        ):
            self.database.import_corpus(
                *self.bundle[:-1],
                (missing_marker,),
            )

    def test_typed_explicit_nongenerated_question_passes(self) -> None:
        result = self.database.import_corpus(
            *self.bundle[:-1],
            (self.expert,),
        )

        self.assertTrue(result["release_id"])

    def test_typed_packaged_legacy_cohort_retains_compatibility(self) -> None:
        parsed = read_and_parse(CORPUS, include_catalog=True)

        result = self.database.import_corpus(*parsed)

        self.assertTrue(result["release_id"])

    def test_typed_exact_legacy_member_can_be_omitted(self) -> None:
        parsed = read_and_parse(CORPUS, include_catalog=True)
        revision_parents = {
            question.revision_of
            for question in parsed[4]
            if question.revision_of is not None
        }
        omitted_id = next(
            question.id
            for question in parsed[4]
            if "generated" not in question.provenance
            and question.id not in revision_parents
        )
        questions = tuple(
            question for question in parsed[4] if question.id != omitted_id
        )

        result = self.database.import_corpus(
            *parsed[:4],
            questions,
            *parsed[5:],
        )

        self.assertTrue(result["release_id"])
        with self.database.read() as connection:
            self.assertIsNone(
                connection.execute(
                    """SELECT 1 FROM release_questions
                       WHERE release_id=? AND question_id=?""",
                    (result["release_id"], omitted_id),
                ).fetchone()
            )

    def test_typed_legacy_content_change_invalidates_manifest(self) -> None:
        parsed = read_and_parse(CORPUS, include_catalog=True)
        questions = list(parsed[4])
        legacy_index = next(
            index
            for index, question in enumerate(questions)
            if "generated" not in question.provenance
        )
        questions[legacy_index] = replace(
            questions[legacy_index],
            stem=questions[legacy_index].stem + " Unattested content mutation.",
        )

        with self.assertRaisesRegex(
            ValidationError,
            "generated_provenance_required",
        ):
            self.database.import_corpus(
                *parsed[:4],
                questions,
                *parsed[5:],
            )

    def test_typed_legacy_lifecycle_change_preserves_compatibility(self) -> None:
        parsed = read_and_parse(CORPUS, include_catalog=True)
        questions = list(parsed[4])
        legacy_index = next(
            index
            for index, question in enumerate(questions)
            if "generated" not in question.provenance
        )
        questions[legacy_index] = replace(
            questions[legacy_index],
            status=QuestionStatus.RETIRED,
        )

        result = self.database.import_corpus(
            *parsed[:4],
            questions,
            *parsed[5:],
        )

        self.assertTrue(result["release_id"])

    def test_typed_legacy_objective_change_invalidates_manifest(self) -> None:
        parsed = read_and_parse(CORPUS, include_catalog=True)
        questions = list(parsed[4])
        legacy_indexes = [
            index
            for index, question in enumerate(questions)
            if "generated" not in question.provenance
            and question.objective is not None
        ]
        target_index = legacy_indexes[0]
        other_objective = next(
            question.objective
            for index, question in enumerate(questions)
            if index != target_index
            and question.objective is not None
            and question.objective.id != questions[target_index].objective_id
        )
        questions[target_index] = replace(
            questions[target_index],
            objective=other_objective,
        )

        with self.assertRaisesRegex(
            ValidationError,
            "generated_provenance_required",
        ):
            self.database.import_corpus(
                *parsed[:4],
                questions,
                *parsed[5:],
            )

    def test_typed_legacy_diagnostic_change_invalidates_manifest(self) -> None:
        parsed = read_and_parse(CORPUS, include_catalog=True)
        questions = list(parsed[4])
        target_index = next(
            index
            for index, question in enumerate(questions)
            if "generated" not in question.provenance
            and any(
                option.diagnostic_objective_id is not None
                for option in question.options
            )
        )
        target = questions[target_index]
        option_index = next(
            index
            for index, option in enumerate(target.options)
            if option.diagnostic_objective_id is not None
        )
        options = list(target.options)
        options[option_index] = replace(
            options[option_index],
            diagnostic_objective_id=(
                options[option_index].diagnostic_objective_id + "_mutated"
            ),
        )
        questions[target_index] = replace(target, options=tuple(options))

        with self.assertRaisesRegex(
            ValidationError,
            "generated_provenance_required",
        ):
            self.database.import_corpus(
                *parsed[:4],
                questions,
                *parsed[5:],
            )

    def test_typed_objects_cannot_bypass_provenance_scalar_types(self) -> None:
        invalid_values = (
            ["not", "an", "object"],
            {"generated": "true", "human_review": True},
        )
        for provenance in invalid_values:
            with self.subTest(provenance=provenance):
                invalid = replace(self.expert, provenance=provenance)
                with self.assertRaisesRegex(ValidationError, "provenance"):
                    self.database.import_corpus(
                        *self.bundle[:-1],
                        (invalid,),
                    )


if __name__ == "__main__":
    unittest.main()
