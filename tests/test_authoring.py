# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import copy
import hashlib
import io
import json
import sqlite3
import tempfile
import threading
import time
import unittest
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from tsq.authoring import (
    AuthoringJobs,
    CoveragePlanner,
    OfflineAuthoringPipeline,
    QuarantineReviewQueue,
    deterministic_test_pipeline,
)
from tsq.cli import command_topics, main
from tsq.corpus import parse_bundle, read_and_parse
from tsq.errors import ConflictError, NotFoundError, ValidationError
from tsq.store import Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"


class FakeGenerator:
    provider_name = "test-provider"
    model_name = "test-model"

    def generate(self, blueprint, source_context):
        misconception_id = blueprint.misconception_ids[0]
        item = {
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
        if blueprint.learning_objective_id is not None:
            item["learning_objective_id"] = blueprint.learning_objective_id
            for option in item["options"]:
                option["diagnostic_objective_id"] = (
                    None if option["correct"] else blueprint.learning_objective_id
                )
        return item


class MissingObjectiveGenerator(FakeGenerator):
    provider_name = "missing-objective-provider"
    model_name = "missing-objective-model"

    def generate(self, blueprint, source_context):
        item = super().generate(blueprint, source_context)
        item.pop("learning_objective_id", None)
        return item


class WrongDiagnosticObjectiveGenerator(FakeGenerator):
    provider_name = "wrong-diagnostic-provider"
    model_name = "wrong-diagnostic-model"

    def generate(self, blueprint, source_context):
        item = super().generate(blueprint, source_context)
        next(option for option in item["options"] if not option["correct"])[
            "diagnostic_objective_id"
        ] = "lo_causal_visibility"
        return item


class BrokenGenerator(FakeGenerator):
    def generate(self, blueprint, source_context):
        return {"id": "broken-generated-item", "status": "approved", "stem": "Too thin"}


class FailingGenerator(FakeGenerator):
    def generate(self, blueprint, source_context):
        raise RuntimeError("deterministic provider failure")


class ContextEchoFailingGenerator(FakeGenerator):
    provider_name = "context-echo-failure-provider"
    model_name = "context-echo-failure-model"

    def generate(self, blueprint, source_context):
        raise RuntimeError(source_context)


class ContextEchoGenerator(FakeGenerator):
    provider_name = "context-echo-provider"
    model_name = "context-echo-model"

    def generate(self, blueprint, source_context):
        item = super().generate(blueprint, source_context)
        item["provider_note"] = source_context
        return item


class CoreContextEchoGenerator(FakeGenerator):
    provider_name = "core-context-echo-provider"
    model_name = "core-context-echo-model"

    def __init__(self, field):
        self.field = field

    def generate(self, blueprint, source_context):
        item = super().generate(blueprint, source_context)
        item["id"] = f"generated-core-echo-{self.field}"
        item["family_id"] = f"f_generated_core_echo_{self.field}"
        if self.field == "stem":
            item["stem"] = source_context
        elif self.field == "option":
            item["options"][0]["text"] = source_context
        elif self.field == "source":
            item["source_ids"][0] = source_context
        elif self.field == "id":
            item["id"] = source_context
        return item


class AuthorityClaimingGenerator(FakeGenerator):
    provider_name = "authority-claim-provider"
    model_name = "authority-claim-model"

    def __init__(self, *, apparently_valid: bool):
        self.apparently_valid = apparently_valid

    def generate(self, blueprint, source_context):
        item = super().generate(blueprint, source_context)
        suffix = "valid" if self.apparently_valid else "malformed"
        item["id"] = f"generated-authority-{suffix}"
        item["family_id"] = f"f_generated_authority_{suffix}"
        activation_review = (
            {
                "reviewer_kind": "human",
                "reviewer_id": "self-declared-human",
                "reviewed_at": "2026-07-24T12:00:00+00:00",
                "independent_of_author": True,
                "attestation_digest": "a" * 64,
            }
            if self.apparently_valid
            else {"reviewer_kind": "model"}
        )
        item["provenance"] = {
            "human_review": True,
            "activation_review": activation_review,
            "activation": "activate_now",
            "review_status": "generator_claims_approval",
            "source_scope": "Untrusted generator-authored source claim.",
        }
        return item


class CountingGenerator(FakeGenerator):
    def __init__(self):
        self.calls = 0
        self.lock = threading.Lock()

    def generate(self, blueprint, source_context):
        with self.lock:
            self.calls += 1
        time.sleep(0.03)
        return super().generate(blueprint, source_context)


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


class InvalidProvenanceGenerator(FakeGenerator):
    provider_name = "invalid-provenance-provider"
    model_name = "invalid-provenance-model"

    def __init__(self, variant):
        self.variant = variant

    def generate(self, blueprint, source_context):
        item = super().generate(blueprint, source_context)
        if self.variant == "cyclic":
            provenance = {}
            provenance["self"] = provenance
        else:
            provenance = {"nonfinite": float("nan")}
        item["provenance"] = provenance
        return item


class ImportabilityAttackGenerator(FakeGenerator):
    provider_name = "importability-attack-provider"
    model_name = "importability-attack-model"

    def __init__(self, variant):
        self.variant = variant

    def generate(self, blueprint, source_context):
        item = super().generate(blueprint, source_context)
        item["id"] = f"generated-importability-{self.variant}"
        item["family_id"] = f"f_generated_importability_{self.variant}"
        if self.variant == "blank_tag":
            item["tags"] = [""]
        elif self.variant == "blank_option_id":
            item["options"][0]["id"] = ""
        elif self.variant == "duplicate_source":
            item["source_ids"].append(item["source_ids"][0])
        elif self.variant == "unknown_revision":
            item["revision_of"] = "q_does_not_exist"
        elif self.variant == "self_revision":
            item["revision_of"] = item["id"]
        return item


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


class ContextEchoReviewer(AcceptingReviewer):
    reviewer_name = "context-echo-reviewer"

    def review(self, item, source_context):
        return {
            "verdict": "accept",
            "independent": True,
            "source_quote": source_context,
        }


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


class BarrierReviewer(AcceptingReviewer):
    reviewer_name = "concurrent-collision-reviewer"

    def __init__(self, barrier):
        self.barrier = barrier

    def review(self, item, source_context):
        self.barrier.wait(timeout=5)
        return super().review(item, source_context)


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
        self.database.import_corpus(
            *read_and_parse(CORPUS, include_catalog=True)
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_coverage_plan_targets_assessable_concepts_not_containers(self) -> None:
        gaps = CoveragePlanner(self.database).gaps(limit=1000)
        concept_ids = {gap.blueprint.concept_id for gap in gaps}
        self.assertNotIn("c_ai_learning_systems", concept_ids)
        self.assertTrue(concept_ids)
        self.assertTrue(all(gap.target_count > gap.current_count for gap in gaps))

    def test_coverage_plan_supports_narrow_operational_filters(self) -> None:
        planner = CoveragePlanner(self.database)
        gaps = planner.gaps(
            limit=1000,
            topic_filter="LLM Agents",
            objective_filter="lo_agent_tool_authorization",
            misconception_filter=(
                "m_agent_tool_availability_is_authorization"
            ),
            goal_filter="objective_misconception_serviceability",
            maximum_difficulty=-0.35,
        )

        self.assertEqual(
            [gap.blueprint.target_difficulty for gap in gaps],
            [-1.0, -0.35],
        )
        self.assertTrue(
            all(
                gap.blueprint.concept_id == "c_agent_tool_use"
                and gap.blueprint.learning_objective_id
                == "lo_agent_tool_authorization"
                and "m_agent_tool_availability_is_authorization"
                in gap.blueprint.misconception_ids
                for gap in gaps
            )
        )
        with self.assertRaisesRegex(ValidationError, "Unknown coverage goal"):
            CoveragePlanner(self.database).gaps(
                goal_filter="inflate_the_corpus"
            )
        for field, value in (
            ("concept_filter", "c_not_in_release"),
            ("objective_filter", "lo_not_in_release"),
            ("misconception_filter", "m_not_in_release"),
        ):
            with self.subTest(field=field):
                with self.assertRaises(NotFoundError):
                    planner.gaps(**{field: value})

        multi_target = next(
            gap
            for gap in planner.gaps(limit=1000)
            if len(gap.blueprint.misconception_ids) > 1
        )
        non_lead = multi_target.blueprint.misconception_ids[-1]
        self.assertNotEqual(
            non_lead, multi_target.blueprint.target_misconception_id
        )
        self.assertIn(
            multi_target,
            planner.gaps(
                limit=1000,
                misconception_filter=non_lead,
            ),
        )

    def test_cli_coverage_filters_are_visible_and_do_not_enqueue(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                [
                    "--db",
                    str(self.database.path),
                    "coverage",
                    "--topic",
                    "LLM Agents",
                    "--objective",
                    "lo_agent_tool_authorization",
                    "--misconception",
                    "m_agent_tool_availability_is_authorization",
                    "--goal",
                    "objective_misconception_serviceability",
                    "--maximum-difficulty",
                    "-0.35",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["gap_count"], 2)
        self.assertEqual(payload["enqueued_job_ids"], [])
        self.assertEqual(payload["filters"]["topic"], "LLM Agents")
        self.assertTrue(
            all(
                gap["blueprint"]["target_difficulty"] <= -0.35
                for gap in payload["gaps"]
            )
        )
        with self.database.read() as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) AS n FROM generation_jobs"
                ).fetchone()["n"],
                0,
            )

    def test_quarantine_queue_exposes_content_bound_review_packets(self) -> None:
        queue = QuarantineReviewQueue(self.database)
        rows = queue.list(
            topic="LLM Agents",
            learning_objective_id="lo_agent_tool_authorization",
        )
        question_id = "q_agent_approval_argument_binding_001"
        self.assertIn(
            question_id,
            {row["question_id"] for row in rows},
        )
        self.assertTrue(
            all(
                not row["runtime_eligible"]
                and row["activation_ceiling"] == "quarantined"
                for row in rows
            )
        )

        detail = queue.show(question_id)
        self.assertEqual(detail["question"]["status"], "quarantined")
        self.assertEqual(detail["release_evidence_weight"], 0.0)
        self.assertEqual(len(detail["question"]["options"]), 4)
        self.assertTrue(
            all(option["rationale"] for option in detail["question"]["options"])
        )
        self.assertTrue(detail["sources"])
        self.assertTrue(detail["concepts"])
        self.assertEqual(
            detail["learning_objective"]["id"],
            "lo_agent_tool_authorization",
        )
        self.assertEqual(
            [row["id"] for row in detail["diagnostic_objectives"]],
            ["lo_agent_tool_authorization"],
        )
        self.assertEqual(
            {row["id"] for row in detail["misconceptions"]},
            {
                option["misconception_id"]
                for option in detail["question"]["options"]
                if option["misconception_id"] is not None
            },
        )
        self.assertTrue(detail["local_family_comparison"])
        self.assertTrue(detail["local_comparison_scope_complete"])
        self.assertTrue(
            detail["local_comparison_scope"]["included_if_same_family"]
        )
        self.assertTrue(
            detail["local_comparison_scope"][
                "same_family_rows_prioritized"
            ]
        )
        self.assertFalse(
            detail["local_comparison_scope"][
                "secondary_concept_only_matches_included"
            ]
        )
        self.assertTrue(
            all(
                comparison["same_family"]
                or comparison["same_learning_objective"]
                or comparison["same_primary_concept"]
                for comparison in detail["local_family_comparison"]
            )
        )
        self.assertIs(
            detail["provenance_claims"]["human_review"],
            False,
        )

        packet = queue.packet(question_id)
        packet_core = dict(packet)
        packet_digest = packet_core.pop("packet_sha256")
        self.assertEqual(packet_digest, canonical_sha256(packet_core))
        self.assertFalse(
            packet["review_contract"]["automatic_activation_permitted"]
        )
        self.assertTrue(
            packet["review_contract"]["human_activation_review_required"]
        )
        self.assertTrue(
            packet["review_contract"][
                "blind_solver_must_not_receive_critic_material"
            ]
        )
        self.assertTrue(
            packet["review_contract"][
                "blind_solver_must_not_receive_family_material"
            ]
        )
        self.assertTrue(
            packet["review_contract"][
                "family_reviewer_must_not_receive_critic_material"
            ]
        )
        self.assertFalse(
            packet["review_contract"][
                "combined_packet_itself_enforces_stage_isolation"
            ]
        )
        self.assertTrue(
            packet["review_contract"]["stage_sequence_advisory_only"]
        )
        self.assertTrue(
            packet["review_contract"]["stage_exports_are_not_access_control"]
        )
        self.assertTrue(
            packet["review_contract"][
                "family_material_family_assignment_fields_blind"
            ]
        )
        self.assertTrue(
            packet["review_contract"][
                "family_material_selection_family_enriched"
            ]
        )
        self.assertFalse(
            packet["review_contract"]["family_independence_claim_resolved"]
        )
        self.assertIn("critic_material", packet)
        forbidden_blind_fields = {
            "correct",
            "rationale",
            "misconception_id",
            "diagnostic_objective_id",
            "provenance",
            "recorded_item_reviews",
        }
        forbidden_family_fields = forbidden_blind_fields | {
            "family_id",
            "independence_note",
            "release_status",
            "same_family",
            "same_learning_objective",
            "same_primary_concept",
            "source_ids",
        }
        self.assertTrue(
            forbidden_blind_fields.isdisjoint(
                set(nested_keys(packet["blind_solver_material"]))
            )
        )
        self.assertTrue(
            forbidden_family_fields.isdisjoint(
                set(nested_keys(packet["family_reviewer_material"]))
            )
        )
        self.assertEqual(packet, queue.packet(question_id))
        blind = queue.packet(question_id, stage="blind")
        family = queue.packet(question_id, stage="family")
        critic = queue.packet(question_id, stage="critic")
        self.assertEqual(
            packet["schema"],
            "tsq-quarantine-review-packet-v3",
        )
        self.assertEqual(blind["stage"], "blind")
        self.assertEqual(family["stage"], "family")
        self.assertEqual(critic["stage"], "critic")
        self.assertEqual(
            {blind["schema"], family["schema"], critic["schema"]},
            {"tsq-quarantine-review-stage-v2"},
        )
        self.assertEqual(
            blind["coordinator_packet_sha256"],
            packet["packet_sha256"],
        )
        self.assertEqual(
            family["coordinator_packet_sha256"],
            packet["packet_sha256"],
        )
        self.assertEqual(
            critic["coordinator_packet_sha256"],
            packet["packet_sha256"],
        )
        self.assertEqual(
            blind["material_sha256"],
            packet["blind_solver_material_sha256"],
        )
        self.assertEqual(
            family["material_sha256"],
            packet["family_reviewer_material_sha256"],
        )
        self.assertEqual(
            critic["material_sha256"],
            packet["critic_material_sha256"],
        )
        self.assertTrue(
            forbidden_blind_fields.isdisjoint(
                set(nested_keys(blind["material"]))
            )
        )
        self.assertTrue(
            forbidden_family_fields.isdisjoint(
                set(nested_keys(family["material"]))
            )
        )
        self.assertNotIn(
            "local_family_comparison",
            blind["material"],
        )
        self.assertTrue(
            family["material"]["comparison_questions"],
        )
        with self.assertRaises(ValidationError):
            queue.packet(question_id, stage="keyed-and-blind")
        with tempfile.TemporaryDirectory() as directory:
            replica = Database(Path(directory) / "replica.db")
            replica.initialize()
            replica.import_corpus(
                *read_and_parse(CORPUS, include_catalog=True)
            )
            replica_queue = QuarantineReviewQueue(replica)
            self.assertEqual(packet, replica_queue.packet(question_id))
            self.assertEqual(
                blind,
                replica_queue.packet(question_id, stage="blind"),
            )
            self.assertEqual(
                family,
                replica_queue.packet(question_id, stage="family"),
            )
        with self.assertRaises(NotFoundError):
            queue.show("q_agent_tool_execution_boundary_001")

    def test_family_review_prioritizes_reused_transformer_family(
        self,
    ) -> None:
        queue = QuarantineReviewQueue(self.database)
        cases = (
            (
                "q_attention_runtime_workspace_boundary_001",
                "f_transformer_sequence_shape_audit",
                "q_transformer_sequence_shape_audit_001",
                "approved",
            ),
            (
                "q_transformer_unexpected_cross_token_path_001",
                "f_transformer_axis_mixing",
                "q_transformer_token_mixing_001",
                "quarantined",
            ),
            (
                "q_generative_prior_shift_001",
                "f_label_shift_prior_odds",
                "q_shift_prior_odds_correction_001",
                "quarantined",
            ),
        )
        forbidden_fields = {
            "correct",
            "rationale",
            "misconception_id",
            "diagnostic_objective_id",
            "provenance",
            "independence_note",
            "recorded_item_reviews",
            "family_id",
            "release_status",
            "same_family",
            "same_learning_objective",
            "same_primary_concept",
            "source_ids",
        }
        for question_id, family_id, neighbor_id, release_status in cases:
            with self.subTest(question_id=question_id):
                detail = queue.show(question_id)
                [same_family] = [
                    comparison
                    for comparison in detail["local_family_comparison"]
                    if comparison["same_family"]
                ]
                self.assertIs(
                    detail["local_family_comparison"][0],
                    same_family,
                )
                self.assertEqual(
                    same_family["question"]["id"],
                    neighbor_id,
                )
                self.assertEqual(
                    same_family["question"]["family_id"],
                    family_id,
                )
                self.assertEqual(
                    same_family["release_status"],
                    release_status,
                )
                self.assertEqual(
                    detail["same_family_comparison_total"],
                    1,
                )
                self.assertEqual(
                    detail["same_family_comparison_included"],
                    1,
                )
                self.assertTrue(
                    detail["same_family_scope_complete"],
                )

                family_packet = queue.packet(
                    question_id,
                    stage="family",
                )
                [redacted_neighbor] = [
                    comparison
                    for comparison in family_packet["material"][
                        "comparison_questions"
                    ]
                    if comparison["question"]["id"] == neighbor_id
                ]
                self.assertEqual(
                    redacted_neighbor["question"]["id"],
                    neighbor_id,
                )
                self.assertTrue(
                    forbidden_fields.isdisjoint(
                        set(nested_keys(family_packet["material"]))
                    )
                )

    def test_family_review_reports_same_family_truncation(self) -> None:
        bundle = json.loads(CORPUS.read_text(encoding="utf-8"))
        candidate = next(
            question
            for question in bundle["questions"]
            if question["id"]
            == "q_attention_runtime_workspace_boundary_001"
        )
        for index in range(100):
            comparison = copy.deepcopy(candidate)
            comparison["id"] = (
                f"q_packet_same_family_fixture_{index:03d}"
            )
            comparison["stem"] = (
                candidate["stem"][:-1]
                + f" Same-family scope fixture {index:03d}?"
            )
            bundle["questions"].append(comparison)
        cross_scope = copy.deepcopy(candidate)
        cross_scope["id"] = "q_aaa_packet_cross_scope_fixture_001"
        cross_scope["stem"] = (
            candidate["stem"][:-1] + " Cross-scope fixture?"
        )
        cross_scope["learning_objective_id"] = (
            "lo_transformer_information_paths"
        )
        cross_scope["concepts"] = [
            {
                "concept_id": "c_transformers",
                "weight": 0.6,
                "role": "primary",
            },
            {
                "concept_id": "c_attention",
                "weight": 0.4,
                "role": "supporting",
            },
        ]
        bundle["questions"].append(cross_scope)

        parsed = parse_bundle(bundle)
        catalog = read_and_parse(CORPUS, include_catalog=True)[5:]
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "comparison.db")
            database.initialize()
            database.import_corpus(*parsed, *catalog)
            queue = QuarantineReviewQueue(database)
            detail = queue.show(candidate["id"])
            packet = queue.packet(candidate["id"])
            family_stage = queue.packet(
                candidate["id"],
                stage="family",
            )

        self.assertEqual(
            detail["same_family_comparison_total"],
            102,
        )
        self.assertEqual(
            detail["same_family_comparison_included"],
            100,
        )
        self.assertFalse(detail["same_family_scope_complete"])
        self.assertFalse(detail["local_comparison_scope_complete"])
        self.assertFalse(
            packet["review_contract"]["same_family_scope_complete"]
        )
        self.assertFalse(
            packet["review_contract"][
                "family_independence_claim_resolved"
            ]
        )
        self.assertNotIn(
            "same_family_scope_complete",
            family_stage["material"],
        )
        self.assertEqual(len(detail["local_family_comparison"]), 100)
        self.assertTrue(
            all(
                comparison["same_family"]
                for comparison in detail["local_family_comparison"]
            )
        )
        cross_scope_detail = next(
            comparison
            for comparison in detail["local_family_comparison"]
            if comparison["question"]["id"] == cross_scope["id"]
        )
        self.assertFalse(
            cross_scope_detail["same_learning_objective"]
        )
        self.assertFalse(cross_scope_detail["same_primary_concept"])

    def test_quarantine_queue_uses_release_topics_and_all_concept_mappings(
        self,
    ) -> None:
        queue = QuarantineReviewQueue(self.database)
        foundation_ids = {
            row["question_id"]
            for row in queue.list(topic="t_ml_foundations", limit=500)
        }
        self.assertEqual(len(foundation_ids), 8)
        self.assertIn(
            "q_conditional_independence_common_cause_001",
            foundation_ids,
        )
        transformer_ids = {
            row["question_id"]
            for row in queue.list(concept_id="c_transformers", limit=500)
        }
        self.assertIn("q_causal_mask_matrix_001", transformer_ids)
        self.assertIn(
            "q_causal_cross_attention_mask_scope_001",
            transformer_ids,
        )
        for kwargs in (
            {"concept_id": "c_not_in_release"},
            {"learning_objective_id": "lo_not_in_release"},
            {"topic": "Topic That Does Not Exist"},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(NotFoundError):
                    queue.list(**kwargs)

    def test_revoked_quarantine_packet_preserves_revocation_ceiling(
        self,
    ) -> None:
        question_id = "q_agent_approval_argument_binding_001"
        reason = "Review found a superseding content concern."
        self.database.revoke_question(question_id, reason)

        queue = QuarantineReviewQueue(self.database)
        listed = next(
            row
            for row in queue.list(topic="LLM Agents")
            if row["question_id"] == question_id
        )
        self.assertTrue(listed["revoked"])
        self.assertEqual(listed["activation_ceiling"], "revoked")
        self.assertEqual(listed["revocation_reason"], reason)

        detail = queue.show(question_id)
        self.assertEqual(detail["activation_ceiling"], "revoked")
        self.assertEqual(detail["revocation"]["reason"], reason)
        packet = queue.packet(question_id)
        self.assertEqual(packet["activation_ceiling"], "revoked")
        self.assertFalse(
            packet["review_contract"]["human_activation_review_required"]
        )
        self.assertTrue(
            packet["review_contract"]["revoked_identity_must_not_be_promoted"]
        )
        self.assertTrue(
            packet["review_contract"][
                "new_immutable_revision_required_if_reconsidered"
            ]
        )

    def test_filtered_cli_enqueue_is_idempotent(self) -> None:
        arguments = [
            "--db",
            str(self.database.path),
            "coverage",
            "--topic",
            "LLM Agents",
            "--objective",
            "lo_agent_tool_authorization",
            "--misconception",
            "m_agent_tool_availability_is_authorization",
            "--goal",
            "objective_misconception_serviceability",
            "--maximum-difficulty",
            "-0.35",
            "--enqueue",
            "--json",
        ]
        payloads = []
        for _ in range(2):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(arguments), 0)
            payloads.append(json.loads(output.getvalue()))
        self.assertEqual(
            payloads[0]["enqueued_job_ids"],
            payloads[1]["enqueued_job_ids"],
        )
        self.assertEqual(len(payloads[0]["enqueued_job_ids"]), 2)
        with self.database.read() as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) AS n FROM generation_jobs"
                ).fetchone()["n"],
                2,
            )

    def test_cli_quarantine_commands_are_read_only(self) -> None:
        question_id = "q_agent_approval_argument_binding_001"
        before = hashlib.sha256(self.database.path.read_bytes()).hexdigest()
        commands = (
            (
                "list",
                "--topic",
                "LLM Agents",
                "--objective",
                "lo_agent_tool_authorization",
            ),
            ("show", question_id),
            ("packet", question_id, "--stage", "combined"),
            ("packet", question_id, "--stage", "blind"),
            ("packet", question_id, "--stage", "family"),
            ("packet", question_id, "--stage", "critic"),
        )
        for command in commands:
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "--db",
                        str(self.database.path),
                        "quarantine",
                        *command,
                        "--json",
                    ]
                )
            self.assertEqual(exit_code, 0)
            json.loads(output.getvalue())
        self.assertEqual(
            hashlib.sha256(self.database.path.read_bytes()).hexdigest(),
            before,
        )

        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                [
                    "--db",
                    str(self.database.path),
                    "quarantine",
                    "packet",
                    question_id,
                    "--stage",
                    "blind",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertIn(question_id, output.getvalue())
        self.assertIn("stage: blind", output.getvalue())

    def test_objective_blueprints_are_release_pinned_and_semantically_complete(self) -> None:
        gap = next(
            gap
            for gap in CoveragePlanner(self.database).gaps(limit=1000)
            if gap.blueprint.learning_objective_id is not None
        )
        blueprint = gap.blueprint
        self.assertIsNotNone(blueprint.corpus_release_id)
        self.assertTrue(blueprint.learning_objective_name)
        self.assertTrue(blueprint.learning_objective_description)
        self.assertIn(
            blueprint.learning_objective_operation,
            {"distinguish", "explain", "predict", "trace", "diagnose", "apply"},
        )
        self.assertEqual(
            blueprint.learning_objective_evidence_type, "selected_response"
        )
        self.assertTrue(blueprint.misconception_ids)
        with self.database.read() as connection:
            mapped_pairs = {
                (row["objective_id"], row["misconception_id"])
                for row in connection.execute(
                    """SELECT DISTINCT diagnostic.objective_id,
                                      option.misconception_id
                       FROM release_option_objectives diagnostic
                       JOIN options option
                         ON option.question_id = diagnostic.question_id
                        AND option.option_id = diagnostic.option_id
                       WHERE diagnostic.release_id = ?""",
                    (blueprint.corpus_release_id,),
                )
            }
        self.assertTrue(
            all(
                (blueprint.learning_objective_id, misconception_id)
                in mapped_pairs
                for misconception_id in blueprint.misconception_ids
            )
        )

    def test_exact_objective_misconception_debt_creates_transfer_job(self) -> None:
        objective_id = "lo_incremental_kv_cache"
        misconception_id = "m_decode_cache_retains_queries"
        self.assertFalse(
            any(
                gap.blueprint.coverage_goal
                == "objective_misconception_serviceability"
                and gap.blueprint.learning_objective_id == objective_id
                and gap.blueprint.target_misconception_id == misconception_id
                for gap in CoveragePlanner(self.database).gaps(limit=1000)
            )
        )
        with self.database.read() as connection:
            release_id = self.database.get_active_release_id(connection)
            trigger = connection.execute(
                """SELECT DISTINCT q.id
                   FROM release_option_objectives diagnostic
                   JOIN options option
                     ON option.question_id = diagnostic.question_id
                    AND option.option_id = diagnostic.option_id
                   JOIN release_questions membership
                     ON membership.release_id = diagnostic.release_id
                    AND membership.question_id = diagnostic.question_id
                   JOIN questions q ON q.id = diagnostic.question_id
                   WHERE diagnostic.release_id = ?
                     AND diagnostic.objective_id = ?
                     AND option.misconception_id = ?
                     AND membership.status IN ('approved', 'calibrated')
                     AND NOT EXISTS (
                         SELECT 1 FROM question_revocations revoked
                         WHERE revoked.question_id = diagnostic.question_id
                     )
                   ORDER BY q.id LIMIT 1""",
                (release_id, objective_id, misconception_id),
            ).fetchone()
        self.assertIsNotNone(trigger)
        self.database.revoke_question(
            trigger["id"], "Objective authoring serviceability regression."
        )

        gaps = CoveragePlanner(self.database).gaps(limit=1000)
        exact = [
            gap
            for gap in gaps
            if gap.blueprint.learning_objective_id == objective_id
            and gap.blueprint.target_misconception_id == misconception_id
        ]
        self.assertEqual(len(exact), 1)
        self.assertEqual((exact[0].current_count, exact[0].target_count), (2, 3))
        self.assertEqual(exact[0].blueprint.kind, "transfer")
        self.assertEqual(
            exact[0].blueprint.coverage_goal,
            "objective_misconception_serviceability",
        )
        self.assertEqual(exact[0].blueprint.misconception_ids[0], misconception_id)
        job_id = CoveragePlanner(self.database).enqueue(exact)[0]
        incomplete = OfflineAuthoringPipeline(
            self.database, FakeGenerator(), (AcceptingReviewer(),)
        ).run_job(job_id, "Approved source excerpt.")
        self.assertFalse(incomplete["accepted_for_reviewed_quarantine"])
        self.assertIn(
            "missing_exact_diagnostic_targets",
            {issue["code"] for issue in incomplete["deterministic_issues"]},
        )

    def test_cross_objective_diagnoses_do_not_count_as_direct_repair_families(self) -> None:
        objective_id = "lo_transformer_information_paths"
        misconception_id = "m_feedforward_layers_mix_token_positions"
        with self.database.read() as connection:
            release_id = self.database.get_active_release_id(connection)
            direct = connection.execute(
                """SELECT DISTINCT q.id
                   FROM release_option_objectives diagnostic
                   JOIN release_question_objectives assessed
                     ON assessed.release_id = diagnostic.release_id
                    AND assessed.question_id = diagnostic.question_id
                   JOIN options option
                     ON option.question_id = diagnostic.question_id
                    AND option.option_id = diagnostic.option_id
                   JOIN release_questions membership
                     ON membership.release_id = diagnostic.release_id
                    AND membership.question_id = diagnostic.question_id
                   JOIN questions q ON q.id = diagnostic.question_id
                   WHERE diagnostic.release_id = ?
                     AND diagnostic.objective_id = ?
                     AND assessed.objective_id = diagnostic.objective_id
                     AND option.misconception_id = ?
                     AND membership.status IN ('approved', 'calibrated')
                     AND NOT EXISTS (
                         SELECT 1 FROM question_revocations revoked
                         WHERE revoked.question_id = diagnostic.question_id
                     )
                   ORDER BY q.id LIMIT 1""",
                (release_id, objective_id, misconception_id),
            ).fetchone()
        self.assertIsNotNone(direct)
        self.database.revoke_question(
            direct["id"], "Cross-objective capacity counting regression."
        )
        exact = next(
            gap
            for gap in CoveragePlanner(self.database).gaps(limit=1000)
            if gap.blueprint.coverage_goal
            == "objective_misconception_serviceability"
            and gap.blueprint.learning_objective_id == objective_id
            and misconception_id in gap.blueprint.misconception_ids
        )
        self.assertEqual((exact.current_count, exact.target_count), (2, 3))

    def test_objective_fixture_round_trips_through_schema_v2_in_quarantine(self) -> None:
        gap = next(
            gap
            for gap in CoveragePlanner(self.database).gaps(limit=1000)
            if gap.blueprint.learning_objective_id is not None
            and gap.blueprint.misconception_ids
        )
        job_id = CoveragePlanner(self.database).enqueue([gap])[0]
        result = deterministic_test_pipeline(self.database).run_job(
            job_id, "Approved objective-specific source material for offline testing."
        )
        item = result["item"]
        self.assertEqual(result["status"], "reviewed")
        self.assertEqual(item["status"], "quarantined")
        self.assertEqual(
            item["learning_objective_id"], gap.blueprint.learning_objective_id
        )
        self.assertTrue(
            all(
                option.get("diagnostic_objective_id")
                == gap.blueprint.learning_objective_id
                for option in item["options"]
                if not option["correct"]
            )
        )

        bundle = json.loads(CORPUS.read_text(encoding="utf-8"))
        bundle["questions"].append(item)
        parsed_questions = parse_bundle(bundle)[4]
        parsed = next(question for question in parsed_questions if question.id == item["id"])
        self.assertEqual(parsed.status.value, "quarantined")
        self.assertEqual(parsed.objective_id, gap.blueprint.learning_objective_id)
        self.assertTrue(
            all(
                option.diagnostic_objective_id == gap.blueprint.learning_objective_id
                for option in parsed.options
                if not option.correct
            )
        )

    def test_objective_artifact_cannot_drop_or_retarget_exact_mappings(self) -> None:
        gap = next(
            gap
            for gap in CoveragePlanner(self.database).gaps(limit=1000)
            if gap.blueprint.learning_objective_id is not None
            and gap.blueprint.learning_objective_id != "lo_causal_visibility"
            and gap.blueprint.misconception_ids
        )
        first_job = CoveragePlanner(self.database).enqueue([gap])[0]
        missing = OfflineAuthoringPipeline(
            self.database, MissingObjectiveGenerator(), (AcceptingReviewer(),)
        ).run_job(first_job, "Approved source excerpt.")
        self.assertFalse(missing["accepted_for_reviewed_quarantine"])
        self.assertIn(
            "blueprint_objective_mismatch",
            {issue["code"] for issue in missing["deterministic_issues"]},
        )

        second_job = CoveragePlanner(self.database).enqueue([gap])[0]
        self.assertNotEqual(first_job, second_job)
        retargeted = OfflineAuthoringPipeline(
            self.database,
            WrongDiagnosticObjectiveGenerator(),
            (AcceptingReviewer(),),
        ).run_job(second_job, "Approved source excerpt.")
        self.assertFalse(retargeted["accepted_for_reviewed_quarantine"])
        self.assertIn(
            "blueprint_diagnostic_objective_mismatch",
            {issue["code"] for issue in retargeted["deterministic_issues"]},
        )

    def test_v1_blueprint_without_objective_fields_still_runs(self) -> None:
        gap = next(
            gap
            for gap in CoveragePlanner(self.database).gaps(limit=1000)
            if gap.blueprint.learning_objective_id is None
            and gap.blueprint.misconception_ids
        )
        payload = {
            key: value
            for key, value in asdict(gap.blueprint).items()
            if key
            in {
                "concept_id",
                "concept_name",
                "kind",
                "target_difficulty",
                "misconception_ids",
                "source_ids",
                "family_constraint",
                "quality_contract",
            }
        }
        job_id = "gen_legacy_blueprint_regression"
        now = datetime.now(timezone.utc).isoformat()
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO generation_jobs(
                       id, blueprint_json, status, prompt_version, created_at, updated_at
                   ) VALUES (?, ?, 'planned', 'item-blueprint-v1', ?, ?)""",
                (job_id, json.dumps(payload), now, now),
            )
        shown = AuthoringJobs(self.database).show(job_id)
        self.assertIsNone(shown["blueprint"]["learning_objective_id"])
        result = OfflineAuthoringPipeline(
            self.database, FakeGenerator(), (AcceptingReviewer(),)
        ).run_job(job_id, "Approved source excerpt for a legacy job.")
        self.assertEqual(result["status"], "reviewed")
        self.assertEqual(result["item"]["status"], "quarantined")

    def test_v2_blueprint_without_release_pin_fails_before_claim(self) -> None:
        gap = next(
            gap
            for gap in CoveragePlanner(self.database).gaps(limit=1000)
            if gap.blueprint.learning_objective_id is None
            and gap.blueprint.misconception_ids
        )
        payload = asdict(gap.blueprint)
        payload["corpus_release_id"] = None
        job_id = "gen_unpinned_v2_regression"
        now = datetime.now(timezone.utc).isoformat()
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO generation_jobs(
                       id, blueprint_json, status, prompt_version, created_at, updated_at
                   ) VALUES (?, ?, 'planned', 'item-blueprint-v2', ?, ?)""",
                (job_id, json.dumps(payload), now, now),
            )
        with self.assertRaisesRegex(ValidationError, "not pinned"):
            OfflineAuthoringPipeline(
                self.database, FakeGenerator(), (AcceptingReviewer(),)
            ).run_job(job_id, "Approved source excerpt.")
        self.assertEqual(AuthoringJobs(self.database).show(job_id)["status"], "planned")

    def test_duplicate_quarantined_family_is_rejected_across_jobs(self) -> None:
        gap = next(
            gap
            for gap in CoveragePlanner(self.database).gaps(limit=1000)
            if gap.blueprint.learning_objective_id is not None
            and gap.blueprint.misconception_ids
        )
        planner = CoveragePlanner(self.database)
        first_job = planner.enqueue([gap])[0]
        first = deterministic_test_pipeline(self.database).run_job(
            first_job, "Stable approved source context."
        )
        self.assertEqual(first["status"], "reviewed")

        second_job = planner.enqueue([gap])[0]
        self.assertNotEqual(first_job, second_job)
        second = deterministic_test_pipeline(self.database).run_job(
            second_job, "Stable approved source context."
        )
        self.assertEqual(second["status"], "rejected")
        codes = {issue["code"] for issue in second["deterministic_issues"]}
        self.assertIn("quarantine_family_collision", codes)
        self.assertIn("quarantine_question_id_collision", codes)

    def test_concurrent_jobs_cannot_certify_the_same_generated_family(self) -> None:
        gap = next(
            gap
            for gap in CoveragePlanner(self.database).gaps(limit=1000)
            if gap.blueprint.learning_objective_id is not None
            and gap.blueprint.misconception_ids
        )
        first_job = CoveragePlanner(self.database).enqueue([gap])[0]
        second_job = "gen_concurrent_duplicate_regression"
        now = datetime.now(timezone.utc).isoformat()
        with self.database.transaction() as connection:
            first = connection.execute(
                "SELECT blueprint_json, prompt_version FROM generation_jobs WHERE id=?",
                (first_job,),
            ).fetchone()
            connection.execute(
                """INSERT INTO generation_jobs(
                       id, blueprint_json, status, prompt_version, created_at, updated_at
                   ) VALUES (?, ?, 'planned', ?, ?, ?)""",
                (
                    second_job,
                    first["blueprint_json"],
                    first["prompt_version"],
                    now,
                    now,
                ),
            )
        barrier = threading.Barrier(2)

        def run(job_id):
            generator = deterministic_test_pipeline(self.database).generator
            return OfflineAuthoringPipeline(
                self.database, generator, (BarrierReviewer(barrier),)
            ).run_job(job_id, "Identical approved source context.")

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(run, (first_job, second_job)))
        self.assertEqual(
            sorted(result["status"] for result in results),
            ["rejected", "reviewed"],
        )
        rejected = next(result for result in results if result["status"] == "rejected")
        codes = {issue["code"] for issue in rejected["deterministic_issues"]}
        self.assertIn("quarantine_family_collision", codes)
        self.assertIn("quarantine_question_id_collision", codes)
        self.assertTrue(self.database.verify_integrity()["ok"])

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
        self.assertTrue(attention[0].blueprint.misconception_ids)
        self.assertNotIn("src_goodfellow_dl_2016", attention[0].blueprint.source_ids)
        self.assertNotEqual(original_sources, attention[0].blueprint.source_ids)

    def test_topics_excludes_revoked_questions_from_direct_counts(self) -> None:
        def topic_count() -> int:
            output = io.StringIO()
            with redirect_stdout(output):
                command_topics(
                    Namespace(db=self.database.path, json=True, concepts=True)
                )
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

    def test_invalid_generator_provenance_is_inertly_rejected(
        self,
    ) -> None:
        for variant in ("cyclic", "nonfinite"):
            with self.subTest(variant=variant):
                gap = next(
                    gap
                    for gap in CoveragePlanner(self.database).gaps(
                        limit=1000
                    )
                    if gap.blueprint.misconception_ids
                )
                job_id = CoveragePlanner(self.database).enqueue([gap])[0]
                result = OfflineAuthoringPipeline(
                    self.database,
                    InvalidProvenanceGenerator(variant),
                    (AcceptingReviewer(),),
                ).run_job(job_id, "approved source excerpt")
                self.assertEqual(result["status"], "rejected")
                self.assertTrue(result["item"]["generator_output_rejected"])
                shown = AuthoringJobs(self.database).show(job_id)
                self.assertEqual(shown["status"], "rejected")
                self.assertEqual(shown["runs"][-1]["status"], "rejected")

    def test_non_importable_generator_fields_are_rejected(self) -> None:
        variants = (
            "blank_tag",
            "blank_option_id",
            "duplicate_source",
            "unknown_revision",
            "self_revision",
        )
        for variant in variants:
            with self.subTest(variant=variant):
                gap = next(
                    gap
                    for gap in CoveragePlanner(self.database).gaps(
                        limit=1000
                    )
                    if gap.blueprint.misconception_ids
                )
                job_id = CoveragePlanner(self.database).enqueue([gap])[0]
                result = OfflineAuthoringPipeline(
                    self.database,
                    ImportabilityAttackGenerator(variant),
                    (AcceptingReviewer(),),
                ).run_job(job_id, "approved source excerpt")
                self.assertEqual(result["status"], "rejected")
                self.assertTrue(result["deterministic_issues"])
                self.assertFalse(
                    result["accepted_for_reviewed_quarantine"]
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

    def test_generator_cannot_self_attest_human_activation(self) -> None:
        for apparently_valid in (False, True):
            with self.subTest(apparently_valid=apparently_valid):
                gap = next(
                    gap
                    for gap in CoveragePlanner(self.database).gaps(
                        limit=1000
                    )
                    if gap.blueprint.misconception_ids
                )
                job_id = CoveragePlanner(self.database).enqueue([gap])[0]
                result = OfflineAuthoringPipeline(
                    self.database,
                    AuthorityClaimingGenerator(
                        apparently_valid=apparently_valid
                    ),
                    (AcceptingReviewer(),),
                ).run_job(job_id, "approved source excerpt")

                self.assertEqual(result["status"], "reviewed")
                provenance = result["item"]["provenance"]
                self.assertIs(provenance["human_review"], False)
                self.assertNotIn("activation_review", provenance)
                self.assertEqual(
                    provenance["human_review_status"],
                    "required_before_activation",
                )
                self.assertEqual(
                    set(
                        provenance[
                            "stripped_generator_authority_fields"
                        ]
                    ),
                    {
                        "activation",
                        "activation_review",
                        "human_review",
                        "review_status",
                    },
                )
                self.assertEqual(
                    len(
                        provenance[
                            "generator_declared_provenance_sha256"
                        ]
                    ),
                    64,
                )

                bundle = json.loads(CORPUS.read_text(encoding="utf-8"))
                bundle["questions"].append(result["item"])
                parsed = parse_bundle(bundle)[4]
                self.assertIn(
                    result["item"]["id"],
                    {question.id for question in parsed},
                )

    def test_exact_source_context_is_redacted_from_persisted_outputs(
        self,
    ) -> None:
        context = "private approved source context unique-marker-94817"
        gap = next(
            gap
            for gap in CoveragePlanner(self.database).gaps(limit=1000)
            if gap.blueprint.misconception_ids
        )
        job_id = CoveragePlanner(self.database).enqueue([gap])[0]
        result = OfflineAuthoringPipeline(
            self.database,
            ContextEchoGenerator(),
            (ContextEchoReviewer(),),
        ).run_job(job_id, context)
        self.assertEqual(result["status"], "rejected")
        self.assertIn(
            "source_context_echo_in_generator_output",
            {
                issue["code"]
                for issue in result["deterministic_issues"]
            },
        )
        self.assertGreater(
            result["exact_source_context_redactions"][
                "generator_output"
            ],
            0,
        )
        self.assertGreater(
            result["exact_source_context_redactions"][
                "reviewer_outputs"
            ],
            0,
        )
        persisted = json.dumps(
            AuthoringJobs(self.database).show(job_id),
            sort_keys=True,
        )
        self.assertNotIn(context, persisted)

    def test_core_item_echo_is_revalidated_after_context_redaction(
        self,
    ) -> None:
        for field in ("stem", "option", "source", "id"):
            with self.subTest(field=field):
                context = (
                    "private exact source material for redaction field "
                    f"{field} unique-marker-75241"
                )
                gap = next(
                    gap
                    for gap in CoveragePlanner(self.database).gaps(
                        limit=1000
                    )
                    if gap.blueprint.misconception_ids
                )
                job_id = CoveragePlanner(self.database).enqueue([gap])[0]
                result = OfflineAuthoringPipeline(
                    self.database,
                    CoreContextEchoGenerator(field),
                    (AcceptingReviewer(),),
                ).run_job(job_id, context)
                self.assertEqual(result["status"], "rejected")
                self.assertIn(
                    "source_context_echo_in_generator_output",
                    {
                        issue["code"]
                        for issue in result["deterministic_issues"]
                    },
                )
                persisted = json.dumps(
                    AuthoringJobs(self.database).show(job_id),
                    sort_keys=True,
                )
                self.assertNotIn(context, persisted)

    def test_provider_exception_cannot_persist_source_context(self) -> None:
        context = "private failure context unique-marker-51729"
        gap = next(
            gap
            for gap in CoveragePlanner(self.database).gaps(limit=1000)
            if gap.blueprint.misconception_ids
        )
        job_id = CoveragePlanner(self.database).enqueue([gap])[0]
        with self.assertRaisesRegex(RuntimeError, context):
            OfflineAuthoringPipeline(
                self.database,
                ContextEchoFailingGenerator(),
                (AcceptingReviewer(),),
            ).run_job(job_id, context)
        persisted = json.dumps(
            AuthoringJobs(self.database).show(job_id),
            sort_keys=True,
        )
        self.assertNotIn(context, persisted)
        self.assertIn(
            "raw exception text and source context were not persisted",
            persisted,
        )

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

    def test_generation_runs_are_claimed_once_and_recorded_immutably(self) -> None:
        gap = next(
            gap
            for gap in CoveragePlanner(self.database).gaps(limit=1000)
            if gap.blueprint.misconception_ids
        )
        job_id = CoveragePlanner(self.database).enqueue([gap])[0]
        generator = CountingGenerator()
        pipeline = OfflineAuthoringPipeline(
            self.database, generator, (AcceptingReviewer(),)
        )

        def execute():
            try:
                pipeline.run_job(job_id, "approved source excerpt")
                return "reviewed"
            except ConflictError:
                return "conflict"

        with ThreadPoolExecutor(max_workers=8) as executor:
            outcomes = list(executor.map(lambda _: execute(), range(8)))

        self.assertEqual(outcomes.count("reviewed"), 1)
        self.assertEqual(outcomes.count("conflict"), 7)
        self.assertEqual(generator.calls, 1)
        job = AuthoringJobs(self.database).show(job_id)
        self.assertEqual(job["status"], "reviewed")
        self.assertEqual(job["run_count"], 1)
        self.assertEqual(job["runs"][0]["attempt"], 1)
        self.assertEqual(job["runs"][0]["status"], "reviewed")
        with self.assertRaises(sqlite3.IntegrityError):
            with self.database.transaction() as connection:
                connection.execute(
                    "UPDATE generation_job_runs SET status='rejected' WHERE job_id=?",
                    (job_id,),
                )
        with self.assertRaises(sqlite3.IntegrityError):
            with self.database.transaction() as connection:
                connection.execute(
                    "DELETE FROM generation_job_runs WHERE job_id=?", (job_id,)
                )

    def test_rejected_job_retry_preserves_prior_attempt(self) -> None:
        gap = next(
            gap
            for gap in CoveragePlanner(self.database).gaps(limit=1000)
            if gap.blueprint.misconception_ids
        )
        job_id = CoveragePlanner(self.database).enqueue([gap])[0]
        first = OfflineAuthoringPipeline(
            self.database, BrokenGenerator(), (AcceptingReviewer(),)
        ).run_job(job_id, "approved source excerpt")
        self.assertEqual(first["status"], "rejected")
        before = AuthoringJobs(self.database).show(job_id)
        first_raw = before["runs"][0]["raw_output"]

        planned = AuthoringJobs(self.database).retry(job_id)
        self.assertEqual(planned["status"], "planned")
        self.assertIsNone(planned["raw_output"])
        second = OfflineAuthoringPipeline(
            self.database, FakeGenerator(), (AcceptingReviewer(),)
        ).run_job(job_id, "approved source excerpt")
        self.assertEqual(second["status"], "reviewed")

        after = AuthoringJobs(self.database).show(job_id)
        self.assertEqual([run["attempt"] for run in after["runs"]], [1, 2])
        self.assertEqual(
            [run["status"] for run in after["runs"]], ["rejected", "reviewed"]
        )
        self.assertEqual(after["runs"][0]["raw_output"], first_raw)
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                main(
                    [
                        "--db",
                        str(self.database.path),
                        "jobs",
                        "show",
                        job_id,
                    ]
                ),
                0,
            )
        rendered = output.getvalue()
        for issue in first["deterministic_issues"]:
            self.assertIn(issue["code"], rendered)
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                main(
                    [
                        "--db",
                        str(self.database.path),
                        "reviews",
                        "show",
                        job_id,
                    ]
                ),
                0,
            )
        rendered_reviews = output.getvalue()
        self.assertIn("attempt 1 [run rejected]", rendered_reviews)
        self.assertIn("attempt 2 [run reviewed]", rendered_reviews)
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_failed_job_requires_explicit_retry(self) -> None:
        gap = next(
            gap
            for gap in CoveragePlanner(self.database).gaps(limit=1000)
            if gap.blueprint.misconception_ids
        )
        job_id = CoveragePlanner(self.database).enqueue([gap])[0]
        with self.assertRaisesRegex(RuntimeError, "deterministic provider failure"):
            OfflineAuthoringPipeline(
                self.database, FailingGenerator(), (AcceptingReviewer(),)
            ).run_job(job_id, "approved source excerpt")
        failed = AuthoringJobs(self.database).show(job_id)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["runs"][0]["status"], "failed")
        self.assertIn("RuntimeError", failed["runs"][0]["error"]["error_type"])
        with self.assertRaises(ConflictError):
            OfflineAuthoringPipeline(
                self.database, FakeGenerator(), (AcceptingReviewer(),)
            ).run_job(job_id, "approved source excerpt")
        retried = AuthoringJobs(self.database).retry(job_id)
        self.assertEqual(retried["status"], "planned")
        self.assertEqual(retried["run_count"], 1)

    def test_cli_can_list_show_run_retry_and_inspect_reviews(self) -> None:
        gap = next(
            gap
            for gap in CoveragePlanner(self.database).gaps(limit=1000)
            if gap.blueprint.misconception_ids
        )
        job_id = CoveragePlanner(self.database).enqueue([gap])[0]
        source_context = Path(self.tempdir.name) / "approved-source.txt"
        source_context.write_text(
            "Approved deterministic test context for offline authoring operations.",
            encoding="utf-8",
        )
        with self.database.read() as connection:
            before_questions = connection.execute(
                "SELECT COUNT(*) AS n FROM questions"
            ).fetchone()["n"]

        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                [
                    "--db",
                    str(self.database.path),
                    "jobs",
                    "run",
                    job_id,
                    "--provider",
                    "deterministic-test",
                    "--source-context",
                    str(source_context),
                    "--json",
                ]
            )
        self.assertEqual(exit_code, 0)
        run_result = json.loads(output.getvalue())
        self.assertEqual(run_result["status"], "reviewed")
        self.assertEqual(run_result["item"]["status"], "quarantined")
        with self.assertRaisesRegex(
            ConflictError,
            "reviewed jobs are terminal",
        ):
            deterministic_test_pipeline(self.database).run_job(
                job_id,
                "Approved context for a forbidden terminal rerun.",
            )

        for arguments, expected_type in (
            (["jobs", "list", "--status", "reviewed", "--json"], list),
            (["jobs", "show", job_id, "--json"], dict),
            (["reviews", "show", job_id, "--json"], dict),
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    ["--db", str(self.database.path), *arguments]
                )
            self.assertEqual(exit_code, 0)
            self.assertIsInstance(json.loads(output.getvalue()), expected_type)

        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                ["--db", str(self.database.path), "jobs", "show", job_id]
            )
        self.assertEqual(exit_code, 0)
        rendered_job = output.getvalue()
        self.assertIn("  validation:", rendered_job)
        for issue in run_result["deterministic_issues"]:
            self.assertIn(
                f"[{issue['severity']}] {issue['code']}: {issue['message']}",
                rendered_job,
            )

        with self.database.read() as connection:
            after_questions = connection.execute(
                "SELECT COUNT(*) AS n FROM questions"
            ).fetchone()["n"]
            generated_question = connection.execute(
                "SELECT 1 FROM questions WHERE id = ?",
                (run_result["item"]["id"],),
            ).fetchone()
        self.assertEqual(after_questions, before_questions)
        self.assertIsNone(generated_question)

        rejected_gap = next(
            gap
            for gap in CoveragePlanner(self.database).gaps(limit=1000)
            if gap.blueprint.misconception_ids
        )
        rejected_job = CoveragePlanner(self.database).enqueue([rejected_gap])[0]
        if rejected_job == job_id:
            # A reviewed job is closed, so the planner must allocate a new one.
            self.fail("Coverage planner reused a closed generation job")
        OfflineAuthoringPipeline(
            self.database, BrokenGenerator(), (AcceptingReviewer(),)
        ).run_job(rejected_job, "approved source excerpt")
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                [
                    "--db",
                    str(self.database.path),
                    "jobs",
                    "retry",
                    rejected_job,
                    "--json",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "planned")

    def test_v5_running_job_migrates_fail_closed_with_history(self) -> None:
        gap = next(
            gap
            for gap in CoveragePlanner(self.database).gaps(limit=1000)
            if gap.blueprint.misconception_ids
        )
        job_id = CoveragePlanner(self.database).enqueue([gap])[0]
        with self.database.transaction() as connection:
            self.database._drop_v6_authoring_triggers(connection)
            connection.execute(
                """UPDATE generation_jobs
                   SET status='running', provider='legacy-provider', model='legacy-model'
                   WHERE id=?""",
                (job_id,),
            )
            connection.execute(
                "UPDATE meta SET value='5' WHERE key='schema_version'"
            )

        self.database.initialize()

        migrated = AuthoringJobs(self.database).show(job_id)
        self.assertEqual(migrated["status"], "failed")
        self.assertEqual(migrated["run_count"], 1)
        self.assertEqual(migrated["runs"][0]["status"], "failed")
        self.assertIn("explicit retry", migrated["runs"][0]["error"]["error"])
        self.assertTrue(self.database.verify_integrity()["ok"])


if __name__ == "__main__":
    unittest.main()
