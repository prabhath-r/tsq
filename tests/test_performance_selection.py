# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from tsq.corpus import read_and_parse
from tsq.cli import main
from tsq.engine import AdaptiveEngine
from tsq.errors import ValidationError
from tsq.evidence import (
    CriterionScale,
    LearningTask,
    RubricCriterion,
    TaskModality,
)
from tsq.performance_ledger import (
    PerformanceLedger,
    PerformanceTaskRelease,
    TaskReleaseReview,
)
from tsq.performance_selection import (
    PRODUCTIVE_PROBE_POLICY_VERSION,
    recommend_performance_tasks,
)
from tsq.store import Database
from tsq.versions import (
    CONCEPT_MODEL_VERSION,
    DEFAULT_LEARNER_MODEL_VERSION,
    OBJECTIVE_GAUSSIAN_MODEL_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"
START = datetime(2114, 2, 3, 9, 0, tzinfo=timezone.utc)
_D0 = "0" * 64
_D1 = "1" * 64
_D2 = "2" * 64


class ProductiveProbeSelectionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.tempdir.name) / "selection.db")
        self.database.initialize()
        self.database.import_corpus(*read_and_parse(CORPUS, include_catalog=True))
        self.engine = AdaptiveEngine(self.database)
        self.engine.create_learner("probe-learner")
        self.session = self.engine.start_session(
            "probe-learner", "t_transformers", seed=113, now=START
        )
        with self.database.read() as connection:
            self.corpus_release_id = connection.execute(
                "SELECT value FROM meta WHERE key='active_corpus_release'"
            ).fetchone()["value"]
            source = connection.execute(
                """SELECT source.id, source.content_hash
                   FROM release_sources membership
                   JOIN sources source ON source.id=membership.source_id
                   WHERE membership.release_id=?
                     AND source.id='src_vaswani_attention_2017'""",
                (self.corpus_release_id,),
            ).fetchone()
        self.source_manifest = ((source["id"], source["content_hash"]),)
        self.objective_task = LearningTask(
            id="task_probe_attention_trace",
            version=1,
            family_id="family_probe_attention_trace",
            title="Trace a broken attention value path",
            modality=TaskModality.DEBUGGING,
            criteria=(
                RubricCriterion(
                    id="criterion_probe_value_route",
                    name="Value-routing diagnosis",
                    scale=CriterionScale.CONTINUOUS,
                    concept_weights=(("c_attention", 1.0),),
                    objective_weights=(("lo_attention_value_routing", 1.0),),
                    dependence_group="probe_value_route",
                ),
            ),
            instructions="Trace the pinned value-routing defect and submit a digest.",
            source_manifests=self.source_manifest,
            administration_id="probe_admin_v1",
            administration_manifest_digest=_D0,
            stimulus_id="probe_attention_stimulus_v1",
            stimulus_digest=_D1,
        )
        self.concept_task = LearningTask(
            id="task_probe_attention_explain",
            version=1,
            family_id="family_probe_attention_explain",
            title="Explain an attention information path",
            modality=TaskModality.EXPLANATION,
            criteria=(
                RubricCriterion(
                    id="criterion_probe_attention_explanation",
                    name="Attention-path explanation",
                    scale=CriterionScale.CONTINUOUS,
                    concept_weights=(("c_attention", 1.0),),
                    dependence_group="probe_attention_explanation",
                ),
            ),
            instructions="Explain the pinned information path and submit a digest.",
            source_manifests=self.source_manifest,
            administration_id="probe_admin_v1",
            administration_manifest_digest=_D0,
            stimulus_id="probe_attention_explanation_v1",
            stimulus_digest=_D2,
        )
        unrelated_task = LearningTask(
            id="task_probe_rag_grounding",
            version=1,
            family_id="family_probe_rag_grounding",
            title="Diagnose unsupported retrieved claims",
            modality=TaskModality.CRITIQUE,
            criteria=(
                RubricCriterion(
                    id="criterion_probe_rag_grounding",
                    name="RAG grounding diagnosis",
                    scale=CriterionScale.CONTINUOUS,
                    concept_weights=(("c_rag_grounding", 1.0),),
                    objective_weights=(("lo_rag_claim_grounding", 1.0),),
                    dependence_group="probe_rag_grounding",
                ),
            ),
            instructions="Classify the pinned claims and submit a critique digest.",
            source_manifests=self.source_manifest,
            administration_id="probe_admin_v1",
            administration_manifest_digest=_D0,
            stimulus_id="probe_rag_stimulus_v1",
            stimulus_digest=_D2,
        )
        bundle = PerformanceTaskRelease(
            title="Productive probe policy fixture",
            corpus_release_id=self.corpus_release_id,
            review=TaskReleaseReview(
                reviewer_kind="human",
                reviewer_id="independent_probe_reviewer",
                reviewed_at=START.isoformat(),
                independent_of_author=True,
                attestation_digest=_D2,
            ),
            tasks=(
                ("approved", self.objective_task),
                ("approved", self.concept_task),
                ("approved", unrelated_task),
            ),
        )
        self.ledger = PerformanceLedger(self.database)
        self.release = self.ledger.publish_release(bundle, now=START)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _publish_tasks(
        self,
        title: str,
        *tasks: LearningTask,
    ) -> dict[str, object]:
        return self.ledger.publish_release(
            PerformanceTaskRelease(
                title=title,
                corpus_release_id=self.corpus_release_id,
                review=TaskReleaseReview(
                    reviewer_kind="human",
                    reviewer_id="independent_probe_reviewer",
                    reviewed_at=START.isoformat(),
                    independent_of_author=True,
                    attestation_digest=_D2,
                ),
                tasks=tuple(("approved", task) for task in tasks),
            ),
            now=START,
        )

    def _projection_boundary(self) -> tuple[int, int, str, int]:
        with self.database.read() as connection:
            learner_revision = connection.execute(
                "SELECT revision FROM learners WHERE id='probe-learner'"
            ).fetchone()["revision"]
            session_revision = connection.execute(
                "SELECT revision FROM sessions WHERE id=?", (self.session["id"],)
            ).fetchone()["revision"]
            projection_hash = self.database.learner_projection_hash(
                "probe-learner", connection
            )
            event_count = connection.execute(
                "SELECT COUNT(*) AS n FROM events"
            ).fetchone()["n"]
        return learner_revision, session_revision, projection_hash, event_count

    def test_exact_objective_probe_is_ranked_and_selection_is_read_only(self) -> None:
        before = self._projection_boundary()

        report = recommend_performance_tasks(
            self.database,
            self.session["id"],
            now=START + timedelta(minutes=1),
        )

        self.assertEqual(report["policy_version"], PRODUCTIVE_PROBE_POLICY_VERSION)
        self.assertEqual(
            report["learner_model_version"],
            DEFAULT_LEARNER_MODEL_VERSION,
        )
        self.assertEqual(
            report["projection_time"],
            (START + timedelta(minutes=1)).isoformat(),
        )
        self.assertEqual(self._projection_boundary(), before)
        self.assertEqual(report["eligible_candidate_count"], 2)
        self.assertEqual(
            report["recommendations"][0]["task_id"],
            self.objective_task.id,
        )
        self.assertEqual(
            report["recommendations"][0]["objective_weights"],
            {"lo_attention_value_routing": 1.0},
        )
        self.assertIn(
            "release_pinned_objective_binding",
            report["recommendations"][0]["reasons"],
        )
        self.assertTrue(report["selection_boundary"]["read_only"])
        self.assertFalse(report["selection_boundary"]["mastery_affected"])
        concept_probe = next(
            item
            for item in report["recommendations"]
            if item["task_id"] == self.concept_task.id
        )
        self.assertEqual(
            concept_probe["components"]["selected_response_uncertainty"],
            1.0,
        )
        self.assertEqual(
            concept_probe["components"][
                "selected_response_evidence_scarcity"
            ],
            1.0,
        )

    def test_fresh_family_is_a_hard_constraint_after_one_probe(self) -> None:
        attempt = self.ledger.start_attempt(
            self.session["id"],
            self.objective_task.id,
            task_version=1,
            task_release_id=self.release["release_id"],
            now=START + timedelta(minutes=1),
        )
        self.ledger.record_action(
            attempt["id"],
            "abandoned",
            {"reason_code": "fixture_exit"},
            now=START + timedelta(minutes=2),
        )

        report = recommend_performance_tasks(
            self.database,
            self.session["id"],
            now=START + timedelta(minutes=3),
        )

        self.assertTrue(report["fresh_family_constraint_applied"])
        self.assertEqual(
            [item["task_id"] for item in report["recommendations"]],
            [self.concept_task.id],
        )
        self.assertEqual(report["recommendations"][0]["prior_family_attempts"], 0)

    def test_unrelated_topic_has_no_eligible_productive_probe(self) -> None:
        self.engine.create_learner("unrelated-probe-learner")
        session = self.engine.start_session(
            "unrelated-probe-learner",
            "t_reinforcement_learning",
            seed=114,
            now=START,
        )

        report = recommend_performance_tasks(
            self.database,
            session["id"],
            now=START + timedelta(minutes=1),
        )

        self.assertEqual(report["eligible_candidate_count"], 0)
        self.assertEqual(report["recommendations"], [])

    def test_cli_recommendation_is_machine_readable_and_non_mutating(self) -> None:
        before = self._projection_boundary()
        output = io.StringIO()
        error = io.StringIO()

        with (
            patch(
                "tsq.performance_selection._now",
                return_value=START + timedelta(minutes=1),
            ),
            redirect_stdout(output),
            redirect_stderr(error),
        ):
            exit_code = main(
                [
                    "--db",
                    str(self.database.path),
                    "task",
                    "recommend",
                    "--session",
                    self.session["id"],
                    "--limit",
                    "1",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0, error.getvalue())
        report = json.loads(output.getvalue())
        self.assertEqual(len(report["recommendations"]), 1)
        self.assertEqual(
            report["recommendations"][0]["task_id"], self.objective_task.id
        )
        self.assertEqual(self._projection_boundary(), before)

    def test_repeated_selection_is_deterministic_and_limit_independent(self) -> None:
        before = self._projection_boundary()
        current_time = START + timedelta(minutes=1)

        first = recommend_performance_tasks(
            self.database,
            self.session["id"],
            limit=1,
            now=current_time,
        )
        repeated = recommend_performance_tasks(
            self.database,
            self.session["id"],
            limit=1,
            now=current_time,
        )
        expanded = recommend_performance_tasks(
            self.database,
            self.session["id"],
            limit=50,
            now=current_time,
        )

        self.assertEqual(repeated, first)
        self.assertEqual(expanded["candidate_digest"], first["candidate_digest"])
        self.assertEqual(
            expanded["eligible_candidate_count"],
            first["eligible_candidate_count"],
        )
        self.assertEqual(self._projection_boundary(), before)

    def test_objective_recommendations_project_retention_at_requested_time(
        self,
    ) -> None:
        reference_task = LearningTask(
            id="task_probe_attention_scaling",
            version=1,
            family_id="family_probe_attention_scaling",
            title="Trace attention resource scaling",
            modality=TaskModality.DEBUGGING,
            criteria=(
                RubricCriterion(
                    id="criterion_probe_attention_scaling",
                    name="Attention resource diagnosis",
                    scale=CriterionScale.CONTINUOUS,
                    concept_weights=(("c_attention", 1.0),),
                    objective_weights=(
                        ("lo_attention_resource_scaling", 1.0),
                    ),
                    dependence_group="probe_attention_scaling",
                ),
            ),
            instructions="Trace the pinned resource boundary and submit a digest.",
            source_manifests=self.source_manifest,
            administration_id="probe_admin_v1",
            administration_manifest_digest=_D0,
            stimulus_id="probe_attention_scaling_v1",
            stimulus_digest=_D1,
        )
        self._publish_tasks(
            "Temporal objective projection fixture",
            reference_task,
        )
        with self.database.transaction() as connection:
            connection.executemany(
                """INSERT INTO objective_states(
                       learner_id, objective_id, mean, variance,
                       stability_hours, exposures, last_seen_at,
                       next_review_at, evidence_mass, as_of_event_id,
                       model_version
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)""",
                (
                    (
                        "probe-learner",
                        "lo_attention_value_routing",
                        3.0,
                        0.1,
                        24.0,
                        4,
                        START.isoformat(),
                        (START + timedelta(days=2)).isoformat(),
                        4.0,
                        OBJECTIVE_GAUSSIAN_MODEL_VERSION,
                    ),
                    (
                        "probe-learner",
                        "lo_attention_resource_scaling",
                        0.0,
                        0.5,
                        24.0,
                        4,
                        None,
                        None,
                        4.0,
                        OBJECTIVE_GAUSSIAN_MODEL_VERSION,
                    ),
                ),
            )
        before = self._projection_boundary()

        near = recommend_performance_tasks(
            self.database,
            self.session["id"],
            limit=50,
            now=START + timedelta(hours=1),
        )
        elapsed = recommend_performance_tasks(
            self.database,
            self.session["id"],
            limit=50,
            now=START + timedelta(days=30),
        )

        def by_id(report: dict[str, object], task_id: str) -> dict[str, object]:
            return next(
                item
                for item in report["recommendations"]  # type: ignore[index]
                if item["task_id"] == task_id
            )

        near_target = by_id(near, self.objective_task.id)
        elapsed_target = by_id(elapsed, self.objective_task.id)
        near_ids = [
            item["task_id"] for item in near["recommendations"]
        ]
        elapsed_ids = [
            item["task_id"] for item in elapsed["recommendations"]
        ]
        self.assertEqual(
            near_target["components"]["due_selected_response_binding_share"],
            0.0,
        )
        self.assertEqual(
            elapsed_target["components"][
                "due_selected_response_binding_share"
            ],
            1.0,
        )
        self.assertGreater(
            elapsed_target["components"]["selected_response_probe_need"],
            near_target["components"]["selected_response_probe_need"],
        )
        self.assertGreater(
            near_ids.index(self.objective_task.id),
            near_ids.index(reference_task.id),
        )
        self.assertLess(
            elapsed_ids.index(self.objective_task.id),
            elapsed_ids.index(reference_task.id),
        )
        self.assertNotEqual(near["candidate_digest"], elapsed["candidate_digest"])
        self.assertEqual(self._projection_boundary(), before)

    def test_concept_recommendations_project_retention_at_requested_time(
        self,
    ) -> None:
        reference_task = LearningTask(
            id="task_probe_causal_masking_explain",
            version=1,
            family_id="family_probe_causal_masking_explain",
            title="Explain a causal masking boundary",
            modality=TaskModality.EXPLANATION,
            criteria=(
                RubricCriterion(
                    id="criterion_probe_causal_masking_explanation",
                    name="Causal-mask explanation",
                    scale=CriterionScale.CONTINUOUS,
                    concept_weights=(("c_causal_masking", 1.0),),
                    dependence_group="probe_causal_masking_explanation",
                ),
            ),
            instructions="Explain the pinned visibility boundary and submit a digest.",
            source_manifests=self.source_manifest,
            administration_id="probe_admin_v1",
            administration_manifest_digest=_D0,
            stimulus_id="probe_causal_masking_explanation_v1",
            stimulus_digest=_D1,
        )
        self._publish_tasks(
            "Temporal concept projection fixture",
            reference_task,
        )
        with self.database.transaction() as connection:
            connection.executemany(
                """INSERT INTO skill_states(
                       learner_id, concept_id, mean, variance,
                       stability_hours, exposures, last_seen_at,
                       next_review_at, evidence_mass, as_of_event_id,
                       model_version
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)""",
                (
                    (
                        "probe-learner",
                        "c_attention",
                        3.0,
                        0.1,
                        24.0,
                        4,
                        START.isoformat(),
                        (START + timedelta(days=2)).isoformat(),
                        4.0,
                        CONCEPT_MODEL_VERSION,
                    ),
                    (
                        "probe-learner",
                        "c_causal_masking",
                        0.0,
                        0.5,
                        24.0,
                        4,
                        None,
                        None,
                        4.0,
                        CONCEPT_MODEL_VERSION,
                    ),
                ),
            )
        before = self._projection_boundary()

        near = recommend_performance_tasks(
            self.database,
            self.session["id"],
            limit=50,
            now=START + timedelta(hours=1),
        )
        elapsed = recommend_performance_tasks(
            self.database,
            self.session["id"],
            limit=50,
            now=START + timedelta(days=30),
        )
        near_by_id = {
            item["task_id"]: item for item in near["recommendations"]
        }
        elapsed_by_id = {
            item["task_id"]: item for item in elapsed["recommendations"]
        }
        near_target = near_by_id[self.concept_task.id]
        elapsed_target = elapsed_by_id[self.concept_task.id]
        near_ids = [
            item["task_id"] for item in near["recommendations"]
        ]
        elapsed_ids = [
            item["task_id"] for item in elapsed["recommendations"]
        ]

        self.assertEqual(
            near_target["components"]["due_selected_response_binding_share"],
            0.0,
        )
        self.assertEqual(
            elapsed_target["components"][
                "due_selected_response_binding_share"
            ],
            1.0,
        )
        self.assertGreater(
            elapsed_target["components"]["selected_response_probe_need"],
            near_target["components"]["selected_response_probe_need"],
        )
        self.assertGreater(
            near_ids.index(self.concept_task.id),
            near_ids.index(reference_task.id),
        )
        self.assertLess(
            elapsed_ids.index(self.concept_task.id),
            elapsed_ids.index(reference_task.id),
        )
        self.assertNotEqual(near["candidate_digest"], elapsed["candidate_digest"])
        self.assertEqual(self._projection_boundary(), before)

    def test_mixed_scope_task_cannot_leak_through_one_matching_criterion(self) -> None:
        mixed_task = LearningTask(
            id="task_probe_mixed_attention_rag",
            version=1,
            family_id="family_probe_mixed_attention_rag",
            title="Diagnose attention and retrieval boundaries",
            modality=TaskModality.DEBUGGING,
            criteria=(
                RubricCriterion(
                    id="criterion_probe_mixed_attention",
                    name="Attention diagnosis",
                    scale=CriterionScale.CONTINUOUS,
                    concept_weights=(("c_attention", 1.0),),
                    objective_weights=(("lo_attention_value_routing", 1.0),),
                    dependence_group="probe_mixed_attention",
                ),
                RubricCriterion(
                    id="criterion_probe_mixed_rag",
                    name="Retrieval grounding diagnosis",
                    scale=CriterionScale.CONTINUOUS,
                    concept_weights=(("c_rag_grounding", 1.0),),
                    objective_weights=(("lo_rag_claim_grounding", 1.0),),
                    dependence_group="probe_mixed_rag",
                ),
            ),
            instructions="Diagnose both pinned boundaries and submit one digest.",
            source_manifests=self.source_manifest,
            administration_id="probe_admin_v1",
            administration_manifest_digest=_D0,
            stimulus_id="probe_mixed_attention_rag_v1",
            stimulus_digest=_D1,
        )
        self._publish_tasks("Mixed-scope containment fixture", mixed_task)

        report = recommend_performance_tasks(
            self.database,
            self.session["id"],
            limit=50,
            now=START + timedelta(minutes=1),
        )

        self.assertNotIn(
            mixed_task.id,
            [item["task_id"] for item in report["recommendations"]],
        )
        self.assertEqual(report["eligible_candidate_count"], 2)

    def test_recommendation_batch_contains_one_task_per_family(self) -> None:
        same_family_task = LearningTask(
            id="task_probe_attention_trace_variant",
            version=1,
            family_id=self.objective_task.family_id,
            title="Trace another attention value path",
            modality=TaskModality.DEBUGGING,
            criteria=self.objective_task.criteria,
            instructions="Trace the alternate pinned defect and submit a digest.",
            source_manifests=self.source_manifest,
            administration_id="probe_admin_v1",
            administration_manifest_digest=_D0,
            stimulus_id="probe_attention_stimulus_variant_v1",
            stimulus_digest=_D2,
        )
        self._publish_tasks("Same-family independence fixture", same_family_task)

        report = recommend_performance_tasks(
            self.database,
            self.session["id"],
            limit=50,
            now=START + timedelta(minutes=1),
        )

        family_ids = [item["family_id"] for item in report["recommendations"]]
        self.assertEqual(len(family_ids), len(set(family_ids)))
        self.assertEqual(report["eligible_candidate_count"], 3)
        self.assertEqual(report["ranked_family_count"], 2)

    def test_partial_objective_binding_retains_its_true_rubric_share(self) -> None:
        partial_task = LearningTask(
            id="task_probe_partial_objective",
            version=1,
            family_id="family_probe_partial_objective",
            title="Trace and explain an attention boundary",
            modality=TaskModality.EXPLANATION,
            criteria=(
                RubricCriterion(
                    id="criterion_probe_partial_objective",
                    name="Objective-specific trace",
                    scale=CriterionScale.CONTINUOUS,
                    concept_weights=(("c_attention", 1.0),),
                    objective_weights=(("lo_attention_value_routing", 1.0),),
                    dependence_group="probe_partial_objective",
                    score_weight=0.01,
                ),
                RubricCriterion(
                    id="criterion_probe_partial_concept",
                    name="Broader concept explanation",
                    scale=CriterionScale.CONTINUOUS,
                    concept_weights=(("c_attention", 1.0),),
                    dependence_group="probe_partial_concept",
                    score_weight=0.99,
                ),
            ),
            instructions="Trace and explain the pinned attention boundary.",
            source_manifests=self.source_manifest,
            administration_id="probe_admin_v1",
            administration_manifest_digest=_D0,
            stimulus_id="probe_partial_objective_v1",
            stimulus_digest=_D1,
        )
        self._publish_tasks("Partial objective-binding fixture", partial_task)

        report = recommend_performance_tasks(
            self.database,
            self.session["id"],
            limit=50,
            now=START + timedelta(minutes=1),
        )
        selected = next(
            item
            for item in report["recommendations"]
            if item["task_id"] == partial_task.id
        )

        self.assertAlmostEqual(
            sum(selected["objective_weights"].values()),
            0.01,
        )
        self.assertAlmostEqual(
            selected["components"]["objective_binding_specificity"],
            0.01,
        )

    def test_sub_rounding_score_difference_is_not_collapsed_into_a_tie(self) -> None:
        def near_tie_task(task_id: str, objective_share: float) -> LearningTask:
            return LearningTask(
                id=task_id,
                version=1,
                family_id=f"family_{task_id}",
                title=f"Near-tie probe {task_id}",
                modality=TaskModality.DESIGN,
                criteria=(
                    RubricCriterion(
                        id=f"criterion_{task_id}_objective",
                        name="Objective-specific design",
                        scale=CriterionScale.CONTINUOUS,
                        concept_weights=(("c_attention", 1.0),),
                        objective_weights=(
                            ("lo_attention_value_routing", 1.0),
                        ),
                        dependence_group=f"dependence_{task_id}_objective",
                        score_weight=objective_share,
                    ),
                    RubricCriterion(
                        id=f"criterion_{task_id}_concept",
                        name="Concept-level design",
                        scale=CriterionScale.CONTINUOUS,
                        concept_weights=(("c_attention", 1.0),),
                        dependence_group=f"dependence_{task_id}_concept",
                        score_weight=1.0 - objective_share,
                    ),
                ),
                instructions="Design the pinned attention path and submit a digest.",
                source_manifests=self.source_manifest,
                administration_id="probe_admin_v1",
                administration_manifest_digest=_D0,
                stimulus_id=f"stimulus_{task_id}",
                stimulus_digest=_D1,
            )

        lower = near_tie_task("task_probe_near_tie_lower", 0.5)
        higher = near_tie_task("task_probe_near_tie_higher", 0.500000001)
        self._publish_tasks("Full-precision ranking fixture", lower, higher)

        report = recommend_performance_tasks(
            self.database,
            self.session["id"],
            limit=50,
            now=START + timedelta(minutes=1),
        )
        by_id = {
            item["task_id"]: item
            for item in report["recommendations"]
        }
        difference = by_id[higher.id]["score"] - by_id[lower.id]["score"]

        self.assertGreater(difference, 0.0)
        self.assertLess(difference, 1e-8)
        self.assertLess(
            [item["task_id"] for item in report["recommendations"]].index(
                higher.id
            ),
            [item["task_id"] for item in report["recommendations"]].index(
                lower.id
            ),
        )

    def test_duplicate_fields_in_stored_task_fail_closed(self) -> None:
        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER performance_tasks_no_update")
            raw = connection.execute(
                """SELECT definition_json FROM performance_tasks
                   WHERE task_id=? AND task_version=1""",
                (self.objective_task.id,),
            ).fetchone()["definition_json"]
            duplicate = (
                raw[:-1]
                + ',"title":'
                + json.dumps(self.objective_task.title)
                + "}"
            )
            connection.execute(
                """UPDATE performance_tasks SET definition_json=?
                   WHERE task_id=? AND task_version=1""",
                (duplicate, self.objective_task.id),
            )

        with self.assertRaisesRegex(ValidationError, "duplicate field 'title'"):
            recommend_performance_tasks(
                self.database,
                self.session["id"],
                now=START + timedelta(minutes=1),
            )

    def test_membership_digest_corruption_fails_closed(self) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                "DROP TRIGGER release_performance_tasks_no_update"
            )
            connection.execute(
                """UPDATE release_performance_tasks SET task_digest=?
                   WHERE release_id=? AND task_id=? AND task_version=1""",
                (_D0, self.release["release_id"], self.objective_task.id),
            )

        with self.assertRaisesRegex(
            ValidationError,
            "membership digest mismatch",
        ):
            recommend_performance_tasks(
                self.database,
                self.session["id"],
                now=START + timedelta(minutes=1),
            )

    def test_incomplete_human_authority_never_becomes_serviceable(self) -> None:
        forged_release_id = "ptrel_incomplete_human"
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO performance_task_releases(
                       id, corpus_release_id, bundle_hash, title,
                       review_json, created_at, sealed_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    forged_release_id,
                    self.corpus_release_id,
                    _D0,
                    "Incomplete human authority",
                    '{"reviewer_kind":"human"}',
                    START.isoformat(),
                    START.isoformat(),
                ),
            )
            connection.execute(
                """INSERT INTO release_performance_tasks(
                       release_id, task_id, task_version, task_digest, status
                   ) VALUES (?, ?, ?, ?, 'pilot')""",
                (
                    forged_release_id,
                    self.objective_task.id,
                    self.objective_task.version,
                    self.objective_task.digest,
                ),
            )
            before_events = connection.execute(
                "SELECT COUNT(*) AS n FROM events"
            ).fetchone()["n"]

        with self.assertRaisesRegex(
            ValidationError,
            "missing|human review|authority",
        ):
            recommend_performance_tasks(
                self.database,
                self.session["id"],
                now=START + timedelta(minutes=1),
            )
        with self.assertRaises(ValidationError):
            self.ledger.start_attempt(
                self.session["id"],
                self.objective_task.id,
                task_version=self.objective_task.version,
                task_release_id=forged_release_id,
                now=START + timedelta(minutes=1),
            )
        with self.database.read() as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) AS n FROM events"
                ).fetchone()["n"],
                before_events,
            )

    def test_invalid_now_type_fails_with_validation_error(self) -> None:
        with self.assertRaisesRegex(ValidationError, "now must be a datetime"):
            recommend_performance_tasks(
                self.database,
                self.session["id"],
                now="2114-02-03T09:01:00Z",  # type: ignore[arg-type]
            )

    def test_pending_question_is_reported_as_a_start_blocker(self) -> None:
        presentation = self.engine.next_question(
            self.session["id"], now=START + timedelta(minutes=1)
        )
        before = self._projection_boundary()

        report = recommend_performance_tasks(
            self.database,
            self.session["id"],
            now=START + timedelta(minutes=2),
        )

        self.assertFalse(report["selection_boundary"]["startable_now"])
        self.assertEqual(
            report["selection_boundary"]["start_blockers"],
            [
                {
                    "code": "pending_question",
                    "id": presentation.decision_id,
                    "resolution": "answer or invalidate the pending question",
                }
            ],
        )
        self.assertEqual(self._projection_boundary(), before)


if __name__ == "__main__":
    unittest.main()
