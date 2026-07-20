# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path

from tsq.authoring import CoveragePlanner, OfflineAuthoringPipeline
from tsq.cli import command_topics
from tsq.corpus import read_and_parse
from tsq.store import Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"


class FakeGenerator:
    provider_name = "test-provider"
    model_name = "test-model"

    def generate(self, blueprint, source_context):
        misconception_id = blueprint.misconception_ids[0]
        return {
            "id": "generated-test-item",
            "family_id": "f_generated_test_independent",
            "status": "approved",
            "stem": (
                f"A practitioner must distinguish the central mechanism of "
                f"{blueprint.concept_name} from a tempting but invalid shortcut. "
                "Which analysis is the most defensible?"
            ),
            "kind": blueprint.kind,
            "difficulty": blueprint.target_difficulty,
            "discrimination": 1.0,
            "guess_rate": 0.25,
            "slip_rate": 0.05,
            "concepts": [
                {"concept_id": blueprint.concept_id, "weight": 1.0, "role": "primary"}
            ],
            "source_ids": [blueprint.source_ids[0]],
            "options": [
                {
                    "id": "a",
                    "text": "Apply the shortcut because its surface resemblance is sufficient evidence for the conclusion.",
                    "correct": False,
                    "misconception_id": misconception_id,
                    "rationale": "Surface resemblance does not establish that the shortcut's assumptions hold in this setting.",
                },
                {
                    "id": "b",
                    "text": "State the governing assumptions, test what is observable, and preserve uncertainty about what is not identified.",
                    "correct": True,
                    "misconception_id": None,
                    "rationale": "The answer separates observed evidence from assumptions and avoids an unsupported identification claim.",
                },
                {
                    "id": "c",
                    "text": "Treat a larger sample as proof that every unobserved assumption has become empirically verified.",
                    "correct": False,
                    "misconception_id": misconception_id,
                    "rationale": "Sampling precision cannot verify structural assumptions that remain unobserved in the data.",
                },
                {
                    "id": "d",
                    "text": "Choose the analysis with the most favorable result because optimization resolves inferential ambiguity.",
                    "correct": False,
                    "misconception_id": misconception_id,
                    "rationale": "Optimizing the reported result does not resolve ambiguity in assumptions or evidence.",
                },
            ],
            "source_context_seen": bool(source_context),
        }


class BrokenGenerator(FakeGenerator):
    def generate(self, blueprint, source_context):
        return {"id": "broken-generated-item", "status": "approved", "stem": "Too thin"}


class CoercionAttackGenerator(FakeGenerator):
    def generate(self, blueprint, source_context):
        item = super().generate(blueprint, source_context)
        # Both values used to be silently accepted by bool()/float().
        item["options"][1]["correct"] = "false"
        item["difficulty"] = str(item["difficulty"])
        return item


class NonObjectGenerator(FakeGenerator):
    def generate(self, blueprint, source_context):
        return ["not", "an", "item"]


class MetadataLeakGenerator(FakeGenerator):
    def generate(self, blueprint, source_context):
        item = super().generate(blueprint, source_context)
        item["answerKey"] = "b"
        item["hidden_review_payload"] = {
            "misconception_labels": ["a", "c", "d"],
            "correctOptionId": "b",
        }
        return item


class AcceptingReviewer:
    reviewer_name = "independent-test-reviewer"

    def review(self, item, source_context):
        return {"verdict": "accept", "independent": True}


class NamedReviewer(AcceptingReviewer):
    def __init__(self, reviewer_name):
        self.reviewer_name = reviewer_name


class SameModelReviewer(AcceptingReviewer):
    reviewer_name = "same-model-under-an-alias"
    provider_name = FakeGenerator.provider_name
    model_name = FakeGenerator.model_name


class DeclaredModelReviewer(NamedReviewer):
    provider_name = "review-provider"
    model_name = "review-model"


class CapturingMutatingReviewer(AcceptingReviewer):
    reviewer_name = "blind-mutation-reviewer"

    def __init__(self):
        self.received = None

    def review(self, item, source_context):
        self.received = copy.deepcopy(item)
        item["stem"] = "A reviewer attempted to replace the generated stem."
        item["options"][0]["text"] = "A reviewer attempted to replace an option."
        item["options"][0]["correct"] = True
        return {"verdict": "accept", "independent": True, "confidence": 0.91}


class InvalidVerdictReviewer(AcceptingReviewer):
    reviewer_name = "invalid-verdict-reviewer"

    def review(self, item, source_context):
        return {"verdict": True, "independent": True}


class CyclicOutputReviewer(AcceptingReviewer):
    reviewer_name = "cyclic-output-reviewer"

    def review(self, item, source_context):
        output = {"verdict": "accept"}
        output["self"] = output
        return output


def canonical_sha256(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def nested_keys(value):
    if isinstance(value, dict):
        yield from value
        for entry in value.values():
            yield from nested_keys(entry)
    elif isinstance(value, list):
        for entry in value:
            yield from nested_keys(entry)


class AuthoringTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.tempdir.name) / "test.db")
        self.database.initialize()
        self.database.import_corpus(*read_and_parse(CORPUS))

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_coverage_plan_targets_assessable_concepts_not_containers(self) -> None:
        gaps = CoveragePlanner(self.database).gaps(limit=1000)
        concept_ids = {gap.blueprint.concept_id for gap in gaps}
        self.assertNotIn("c_ai_learning_systems", concept_ids)
        self.assertTrue(concept_ids)
        self.assertTrue(all(gap.target_count > gap.current_count for gap in gaps))

    def test_coverage_plan_excludes_revoked_items_from_all_live_evidence(self) -> None:
        planner = CoveragePlanner(self.database)
        initial = planner.gaps(limit=1000)
        attention = [
            gap
            for gap in initial
            if gap.blueprint.concept_id == "c_attention"
            and gap.blueprint.kind == "conceptual"
        ]
        self.assertEqual(len(attention), 1)
        self.assertEqual(attention[0].current_count, 1)
        original_sources = attention[0].blueprint.source_ids
        self.assertIn("src_goodfellow_dl_2016", original_sources)

        self.database.revoke_question(
            "q_attention_sequence_scaling_001",
            "Coverage regression: item is no longer selectable.",
        )

        updated = planner.gaps(limit=1000)
        attention = [
            gap
            for gap in updated
            if gap.blueprint.concept_id == "c_attention"
            and gap.blueprint.kind == "conceptual"
        ]
        self.assertEqual(len(attention), 2)
        self.assertTrue(all(gap.current_count == 0 for gap in attention))
        self.assertIn(
            "m_quadratic_attention_only_during_training",
            attention[0].blueprint.misconception_ids,
        )
        self.assertNotIn("src_goodfellow_dl_2016", attention[0].blueprint.source_ids)
        self.assertNotEqual(original_sources, attention[0].blueprint.source_ids)

    def test_topics_excludes_revoked_questions_from_direct_counts(self) -> None:
        def topic_count() -> int:
            output = io.StringIO()
            with redirect_stdout(output):
                command_topics(Namespace(db=self.database.path, json=True))
            rows = json.loads(output.getvalue())
            return next(
                row["direct_questions"]
                for row in rows
                if row["id"] == "c_adaptive_testing"
            )

        initial_count = topic_count()
        self.assertGreaterEqual(initial_count, 1)
        self.database.revoke_question(
            "q_adaptive_item_selection_001",
            "Topics regression: item is no longer selectable.",
        )
        self.assertEqual(topic_count(), initial_count - 1)

    def test_generated_item_is_forced_into_quarantine(self) -> None:
        planner = CoveragePlanner(self.database)
        gap = next(gap for gap in planner.gaps(limit=1000) if gap.blueprint.misconception_ids)
        job_id = planner.enqueue([gap])[0]
        pipeline = OfflineAuthoringPipeline(
            self.database, FakeGenerator(), (AcceptingReviewer(),)
        )
        result = pipeline.run_job(job_id, "approved source excerpt")
        self.assertEqual(result["item"]["status"], "quarantined")
        self.assertTrue(result["accepted_by_critics"])
        self.assertTrue(result["accepted_for_reviewed_quarantine"])
        with self.database.read() as connection:
            row = connection.execute(
                "SELECT status, raw_output_json FROM generation_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        self.assertEqual(row["status"], "reviewed")
        self.assertIn('"status": "quarantined"', row["raw_output_json"])

    def test_critic_acceptance_cannot_bypass_deterministic_validation(self) -> None:
        planner = CoveragePlanner(self.database)
        gap = next(gap for gap in planner.gaps(limit=1000) if gap.blueprint.misconception_ids)
        job_id = planner.enqueue([gap])[0]
        result = OfflineAuthoringPipeline(
            self.database, BrokenGenerator(), (AcceptingReviewer(),)
        ).run_job(job_id, "approved source excerpt")
        self.assertTrue(result["accepted_by_critics"])
        self.assertFalse(result["accepted_for_reviewed_quarantine"])
        self.assertTrue(result["deterministic_issues"])
        with self.database.read() as connection:
            status = connection.execute(
                "SELECT status FROM generation_jobs WHERE id = ?", (job_id,)
            ).fetchone()["status"]
        self.assertEqual(status, "rejected")

    def test_enqueue_is_idempotent_for_an_open_blueprint(self) -> None:
        planner = CoveragePlanner(self.database)
        gap = planner.gaps(limit=1)[0]
        first = planner.enqueue([gap])
        second = planner.enqueue([gap])
        self.assertEqual(first, second)
        with self.database.read() as connection:
            count = connection.execute(
                "SELECT COUNT(*) AS n FROM generation_jobs"
            ).fetchone()["n"]
        self.assertEqual(count, 1)

    def test_generator_scalars_are_never_coerced_into_valid_item_types(self) -> None:
        planner = CoveragePlanner(self.database)
        gap = next(gap for gap in planner.gaps(limit=1000) if gap.blueprint.misconception_ids)
        job_id = planner.enqueue([gap])[0]
        result = OfflineAuthoringPipeline(
            self.database, CoercionAttackGenerator(), (AcceptingReviewer(),)
        ).run_job(job_id, "approved source excerpt")
        messages = " ".join(issue["message"] for issue in result["deterministic_issues"])
        self.assertIn("options[1].correct must be a JSON boolean", messages)
        self.assertIn("'difficulty' must be a finite JSON number", messages)
        self.assertFalse(result["accepted_for_reviewed_quarantine"])
        with self.database.read() as connection:
            status = connection.execute(
                "SELECT status FROM generation_jobs WHERE id = ?", (job_id,)
            ).fetchone()["status"]
        self.assertEqual(status, "rejected")

    def test_non_object_generator_output_is_rejected_without_stringification(self) -> None:
        planner = CoveragePlanner(self.database)
        gap = planner.gaps(limit=1)[0]
        job_id = planner.enqueue([gap])[0]
        result = OfflineAuthoringPipeline(
            self.database, NonObjectGenerator(), (AcceptingReviewer(),)
        ).run_job(job_id, "approved source excerpt")
        self.assertFalse(result["accepted_for_reviewed_quarantine"])
        self.assertEqual(result["item"]["generator_output_rejected"], True)
        self.assertIn(
            "Generated item must be a JSON object",
            result["deterministic_issues"][0]["message"],
        )

    def test_reviewers_receive_blinded_isolated_copies(self) -> None:
        planner = CoveragePlanner(self.database)
        gap = next(gap for gap in planner.gaps(limit=1000) if gap.blueprint.misconception_ids)
        job_id = planner.enqueue([gap])[0]
        reviewer = CapturingMutatingReviewer()
        result = OfflineAuthoringPipeline(
            self.database, MetadataLeakGenerator(), (reviewer,)
        ).run_job(job_id, "approved source excerpt")

        forbidden = {
            "correct",
            "rationale",
            "misconception_id",
            "misconception_ids",
            "answer",
            "answer_key",
        }
        self.assertTrue(forbidden.isdisjoint(set(nested_keys(reviewer.received))))
        self.assertNotIn("provenance", reviewer.received)
        self.assertNotIn("status", reviewer.received)
        self.assertNotIn("answerKey", reviewer.received)
        self.assertNotIn("hidden_review_payload", reviewer.received)
        self.assertNotEqual(
            result["item"]["stem"],
            "A reviewer attempted to replace the generated stem.",
        )
        self.assertNotEqual(
            result["item"]["options"][0]["text"],
            "A reviewer attempted to replace an option.",
        )
        self.assertTrue(result["accepted_for_reviewed_quarantine"])

    def test_reviewer_identities_must_be_unique_and_distinct_from_generator(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate reviewer identity"):
            OfflineAuthoringPipeline(
                self.database,
                FakeGenerator(),
                (NamedReviewer("critic-a"), NamedReviewer(" CRITIC-A ")),
            )
        with self.assertRaisesRegex(ValueError, "collides with the generator"):
            OfflineAuthoringPipeline(
                self.database,
                FakeGenerator(),
                (NamedReviewer(FakeGenerator.model_name),),
            )
        with self.assertRaisesRegex(ValueError, "same provider/model identity"):
            OfflineAuthoringPipeline(
                self.database,
                FakeGenerator(),
                (SameModelReviewer(),),
            )
        with self.assertRaisesRegex(ValueError, "same provider/model identity"):
            OfflineAuthoringPipeline(
                self.database,
                FakeGenerator(),
                (DeclaredModelReviewer("critic-a"), DeclaredModelReviewer("critic-b")),
            )

    def test_source_and_review_attestations_are_hashed_and_persisted(self) -> None:
        planner = CoveragePlanner(self.database)
        gap = next(gap for gap in planner.gaps(limit=1000) if gap.blueprint.misconception_ids)
        job_id = planner.enqueue([gap])[0]
        context = "private approved source excerpt with a unique marker 7429"
        result = OfflineAuthoringPipeline(
            self.database, FakeGenerator(), (AcceptingReviewer(),)
        ).run_job(job_id, context)
        expected_context_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()
        self.assertEqual(result["source_context_sha256"], expected_context_hash)

        with self.database.read() as connection:
            row = connection.execute(
                "SELECT raw_output_json, validation_json FROM generation_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        persisted_item = json.loads(row["raw_output_json"])
        persisted = json.loads(row["validation_json"])
        review = persisted["reviews"][0]
        self.assertEqual(persisted["source_context_sha256"], expected_context_hash)
        self.assertEqual(
            persisted_item["provenance"]["source_context_sha256"], expected_context_hash
        )
        self.assertEqual(
            review["reviewer_output_sha256"], canonical_sha256(review["output"])
        )
        self.assertEqual(
            review["reviewer_provenance_sha256"], canonical_sha256(review["reviewer"])
        )
        self.assertEqual(persisted["reviews_sha256"], canonical_sha256(persisted["reviews"]))
        self.assertEqual(
            persisted["generator_provenance_sha256"],
            canonical_sha256(persisted["generator_provenance"]),
        )
        self.assertNotIn(context, row["raw_output_json"])
        self.assertNotIn(context, row["validation_json"])

    def test_non_string_reviewer_verdict_cannot_authorize_quarantine(self) -> None:
        planner = CoveragePlanner(self.database)
        gap = next(gap for gap in planner.gaps(limit=1000) if gap.blueprint.misconception_ids)
        job_id = planner.enqueue([gap])[0]
        result = OfflineAuthoringPipeline(
            self.database, FakeGenerator(), (InvalidVerdictReviewer(),)
        ).run_job(job_id, "approved source excerpt")
        self.assertFalse(result["accepted_by_critics"])
        self.assertFalse(result["accepted_for_reviewed_quarantine"])
        self.assertFalse(result["reviews"][0]["valid"])
        self.assertTrue(result["reviews"][0]["validation_errors"])

    def test_cyclic_reviewer_output_is_safely_rejected(self) -> None:
        planner = CoveragePlanner(self.database)
        gap = next(gap for gap in planner.gaps(limit=1000) if gap.blueprint.misconception_ids)
        job_id = planner.enqueue([gap])[0]
        result = OfflineAuthoringPipeline(
            self.database, FakeGenerator(), (CyclicOutputReviewer(),)
        ).run_job(job_id, "approved source excerpt")
        self.assertFalse(result["accepted_by_critics"])
        self.assertIn("cyclic JSON object", result["reviews"][0]["validation_errors"][0])
        with self.database.read() as connection:
            row = connection.execute(
                "SELECT status, validation_json FROM generation_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        self.assertEqual(row["status"], "rejected")
        json.loads(row["validation_json"])


if __name__ == "__main__":
    unittest.main()
