# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tsq.cli import main
from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.errors import ValidationError
from tsq.evidence import (
    ActionPhase,
    CriterionScale,
    EvaluationStatus,
    LearningTask,
    RubricCriterion,
    TaskModality,
    canonical_digest,
    canonical_json,
)
from tsq.performance import ImportedCriterionResult, ImportedEvaluation
from tsq.performance_ledger import (
    PerformanceLedger,
    PerformanceTaskRelease,
    TaskReleaseReview,
)
from tsq.store import Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"
START = datetime(2112, 8, 9, 10, 0, tzinfo=timezone.utc)
_D0 = "0" * 64
_D1 = "1" * 64
_D2 = "2" * 64
_D3 = "3" * 64
_D4 = "4" * 64


class PerformanceReportingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.tempdir.name) / "report.db")
        self.database.initialize()
        self.database.import_corpus(
            *read_and_parse(CORPUS, include_catalog=True)
        )
        self.engine = AdaptiveEngine(self.database)
        self.engine.create_learner("report-learner")
        self.session = self.engine.start_session(
            "report-learner",
            "t_transformers",
            seed=81,
            now=START,
        )
        with self.database.read() as connection:
            corpus_release_id = connection.execute(
                "SELECT value FROM meta WHERE key='active_corpus_release'"
            ).fetchone()["value"]
            source = connection.execute(
                """SELECT source.id, source.content_hash
                   FROM release_sources membership
                   JOIN sources source ON source.id=membership.source_id
                   WHERE membership.release_id=?
                     AND source.id='src_vaswani_attention_2017'""",
                (corpus_release_id,),
            ).fetchone()
        self.task = LearningTask(
            id="task_reporting_attention_debug",
            version=1,
            family_id="family_reporting_attention_debug",
            title="Debug a causal attention mask",
            modality=TaskModality.DEBUGGING,
            criteria=(
                RubricCriterion(
                    id="criterion_reporting_mask",
                    name="Causal mask invariant",
                    scale=CriterionScale.CONTINUOUS,
                    concept_weights=(("c_causal_masking", 1.0),),
                    objective_weights=(
                        ("lo_causal_visibility", 1.0),
                    ),
                    dependence_group="reporting_mask_behavior",
                    misconception_ids=(
                        "m_mask_only_inference",
                    ),
                    evidence_cap=0.8,
                    dependence_cap=0.8,
                ),
            ),
            instructions=(
                "Inspect the pinned mask stimulus and submit a "
                "content-addressed repair."
            ),
            source_manifests=((source["id"], source["content_hash"]),),
            administration_id="reporting_admin_v1",
            administration_manifest_digest=_D0,
            stimulus_id="reporting_mask_stimulus_v1",
            stimulus_digest=_D1,
        )
        release = PerformanceTaskRelease(
            title="Reporting integration fixture",
            corpus_release_id=corpus_release_id,
            review=TaskReleaseReview(
                reviewer_kind="human",
                reviewer_id="independent_reporting_reviewer",
                reviewed_at=START.isoformat(),
                independent_of_author=True,
                attestation_digest=_D2,
            ),
            tasks=(("pilot", self.task),),
        )
        self.ledger = PerformanceLedger(self.database)
        self.release = self.ledger.publish_release(release, now=START)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _complete_attempt(self) -> str:
        attempt = self.ledger.start_attempt(
            self.session["id"],
            self.task.id,
            task_version=self.task.version,
            task_release_id=self.release["release_id"],
            now=START + timedelta(minutes=1),
        )
        self.ledger.record_action(
            attempt["id"],
            "hint_requested",
            {"hint_id": "mask_hint", "level": 1},
            phase="assisted",
            now=START + timedelta(minutes=2),
        )
        self.ledger.record_action(
            attempt["id"],
            "check_run",
            {
                "check_set_id": "mask_checks_v1",
                "passed": 3,
                "failed": 1,
                "errored": 0,
                "skipped": 0,
                "result_digest": _D3,
            },
            phase="assisted",
            now=START + timedelta(minutes=4),
        )
        submitted = self.ledger.record_action(
            attempt["id"],
            "submitted",
            {"submission_digest": _D4},
            phase="assisted",
            now=START + timedelta(minutes=6),
        )
        self.ledger.import_evaluation(
            attempt["id"],
            ImportedEvaluation(
                criteria=(
                    ImportedCriterionResult(
                        criterion_id="criterion_reporting_mask",
                        status=EvaluationStatus.VALID,
                        score=0.75,
                        outcome_code="partial_mask_repair",
                        phase=ActionPhase.ASSISTED,
                        source_action_ids=(submitted["id"],),
                        misconception_ids=(
                            "m_mask_only_inference",
                        ),
                        reliability=0.8,
                    ),
                )
            ),
            provider_id="reporting_import",
            provider_version="v1",
            now=START + timedelta(minutes=7),
        )
        return attempt["id"]

    def test_empty_reports_expose_an_explicit_zero_shadow_contract(self) -> None:
        profile = self.engine.profile(
            "report-learner", now=START + timedelta(minutes=1)
        )
        session = self.engine.session_report(
            self.session["id"], now=START + timedelta(minutes=1)
        )

        for report in (profile, session):
            shadow = report["productive_skill_shadow"]
            self.assertEqual(shadow["attempt_count"], 0)
            self.assertTrue(shadow["shadow_only"])
            self.assertFalse(shadow["evidence_boundary"]["mastery_claim"])
            self.assertFalse(
                shadow["evidence_boundary"][
                    "selected_response_projection_affected"
                ]
            )
            self.assertFalse(
                shadow["evidence_boundary"][
                    "declared_objective_bindings_applied"
                ]
            )

    def test_shadow_metrics_surface_without_changing_mastery_or_routing(self) -> None:
        with self.database.read() as connection:
            learner_revision_before = connection.execute(
                "SELECT revision FROM learners WHERE id='report-learner'"
            ).fetchone()["revision"]
            projection_hash_before = self.database.learner_projection_hash(
                "report-learner", connection
            )

        attempt_id = self._complete_attempt()
        session_report = self.engine.session_report(
            self.session["id"], now=START + timedelta(minutes=8)
        )
        profile = self.engine.profile(
            "report-learner", now=START + timedelta(minutes=8)
        )

        shadow = session_report["productive_skill_shadow"]
        self.assertEqual(shadow["attempt_count"], 1)
        self.assertEqual(shadow["attempt_statuses"], {"submitted": 1})
        self.assertEqual(shadow["modalities"], {"debugging": 1})
        self.assertEqual(shadow["observed_elapsed_seconds"], 300.0)
        self.assertEqual(shadow["behavior"]["actions"], 4)
        self.assertEqual(shadow["behavior"]["hint_requests"], 1)
        self.assertEqual(shadow["behavior"]["check_runs"], 1)
        self.assertEqual(shadow["rubric_observations"]["evaluations"], 1)
        self.assertEqual(
            shadow["rubric_observations"]["by_status"], {"valid": 1}
        )
        self.assertEqual(
            shadow["rubric_observations"]["valid_score_average"], 0.75
        )
        self.assertEqual(
            shadow["rubric_observations"]["misconception_signals"],
            {"m_mask_only_inference": 1},
        )
        self.assertEqual(
            shadow["scope_binding"]["objective_ids"],
            ["lo_causal_visibility"],
        )
        self.assertTrue(
            shadow["scope_binding"]["objective_binding_available"]
        )
        self.assertEqual(
            shadow["scope_binding"]["objective_bindings"],
            [
                {
                    "task_id": "task_reporting_attention_debug",
                    "task_version": 1,
                    "criterion_id": "criterion_reporting_mask",
                    "objective_weights": [
                        {
                            "objective_id": "lo_causal_visibility",
                            "weight": 1.0,
                        }
                    ],
                }
            ],
        )
        self.assertEqual(
            shadow["rubric_observations"]["by_objective"][
                "lo_causal_visibility"
            ]["valid_score_average"],
            0.75,
        )
        self.assertEqual(shadow["recent_attempts"][0]["attempt_id"], attempt_id)
        self.assertFalse(shadow["recent_attempts"][0]["mastery_claim"])
        self.assertEqual(
            profile["productive_skill_shadow"]["attempt_count"], 1
        )
        self.assertTrue(
            all(
                row["evidence_mass"] == 0
                for row in profile["learning_objectives"]
            )
        )
        with self.database.read() as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT revision FROM learners WHERE id='report-learner'"
                ).fetchone()["revision"],
                learner_revision_before,
            )
            self.assertEqual(
                self.database.learner_projection_hash(
                    "report-learner", connection
                ),
                projection_hash_before,
            )

    def test_topic_scoped_profile_excludes_unrelated_task_families(self) -> None:
        self._complete_attempt()

        transformer_profile = self.engine.profile(
            "report-learner",
            root_concept_id="t_transformers",
            now=START + timedelta(minutes=8),
        )
        unrelated_profile = self.engine.profile(
            "report-learner",
            root_concept_id="t_causal_inference",
            now=START + timedelta(minutes=8),
        )

        self.assertEqual(
            transformer_profile["productive_skill_shadow"]["attempt_count"],
            1,
        )
        self.assertEqual(
            unrelated_profile["productive_skill_shadow"]["attempt_count"],
            0,
        )

    def test_cli_labels_productive_observations_as_non_mastery(self) -> None:
        self._complete_attempt()
        for command in (
            ["session", "report", self.session["id"]],
            ["profile", "--learner", "report-learner"],
        ):
            output = io.StringIO()
            error = io.StringIO()
            with redirect_stdout(output), redirect_stderr(error):
                exit_code = main(
                    ["--db", str(self.database.path), *command]
                )
            self.assertEqual(exit_code, 0, error.getvalue())
            rendered = output.getvalue()
            self.assertIn("productive-task shadow observations", rendered)
            self.assertIn("no mastery update", rendered)
            self.assertIn("raw valid-score mean 75.0%", rendered)

    def _assert_profile_fails_closed(self, pattern: str) -> None:
        with self.assertRaisesRegex(ValidationError, pattern):
            self.engine.profile(
                "report-learner",
                now=START + timedelta(minutes=8),
            )

    def test_report_rejects_noncanonical_task_definition(self) -> None:
        self._complete_attempt()
        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER performance_tasks_no_update")
            connection.execute(
                """UPDATE performance_tasks
                   SET definition_json=definition_json || ' '
                   WHERE task_id=? AND task_version=?""",
                (self.task.id, self.task.version),
            )

        self._assert_profile_fails_closed("canonical stored representation")

    def test_report_rejects_task_digest_commitment_drift(self) -> None:
        self._complete_attempt()
        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER performance_tasks_no_update")
            connection.execute(
                """UPDATE performance_tasks SET task_digest=?
                   WHERE task_id=? AND task_version=?""",
                (_D4, self.task.id, self.task.version),
            )

        self._assert_profile_fails_closed("digest commitment")

    def test_report_rejects_unknown_action_kind(self) -> None:
        attempt_id = self._complete_attempt()
        with self.database.transaction() as connection:
            connection.execute("PRAGMA ignore_check_constraints=ON")
            connection.execute("DROP TRIGGER performance_actions_no_update")
            connection.execute(
                """UPDATE performance_actions SET action_type='unknown_action'
                   WHERE attempt_id=? AND sequence=0""",
                (attempt_id,),
            )

        self._assert_profile_fails_closed(
            "Performance action .* cannot be reported safely"
        )

    def test_report_rejects_noncanonical_action_payload(self) -> None:
        attempt_id = self._complete_attempt()
        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER performance_actions_no_update")
            connection.execute(
                """UPDATE performance_actions SET payload_json=' { }'
                   WHERE attempt_id=? AND sequence=0""",
                (attempt_id,),
            )

        self._assert_profile_fails_closed("canonical stored representation")

    def test_report_rejects_server_elapsed_time_drift(self) -> None:
        attempt_id = self._complete_attempt()
        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER performance_actions_no_update")
            connection.execute(
                """UPDATE performance_actions SET elapsed_ms=elapsed_ms + 1
                   WHERE attempt_id=? AND sequence=1""",
                (attempt_id,),
            )

        self._assert_profile_fails_closed(
            "relational projection|server-derived elapsed time"
        )

    def test_report_rejects_noncanonical_evaluation_and_digest_drift(
        self,
    ) -> None:
        attempt_id = self._complete_attempt()
        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER task_evaluations_no_update")
            connection.execute(
                """UPDATE task_evaluations
                   SET evaluation_json=evaluation_json || ' '
                   WHERE attempt_id=?""",
                (attempt_id,),
            )

        self._assert_profile_fails_closed("canonical stored representation")

    def test_report_rejects_evaluation_digest_drift(self) -> None:
        attempt_id = self._complete_attempt()
        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER task_evaluations_no_update")
            connection.execute(
                """UPDATE task_evaluations SET evaluation_digest=?
                   WHERE attempt_id=?""",
                (_D0, attempt_id),
            )

        self._assert_profile_fails_closed("task commitments")

    def test_report_rejects_coherently_forged_authority_decision(self) -> None:
        attempt_id = self._complete_attempt()
        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER task_evaluations_no_update")
            connection.execute("DROP TRIGGER events_no_update")
            evaluation_row = connection.execute(
                """SELECT * FROM task_evaluations
                   WHERE attempt_id=?""",
                (attempt_id,),
            ).fetchone()
            event = connection.execute(
                "SELECT * FROM events WHERE event_id=?",
                (evaluation_row["event_id"],),
            ).fetchone()
            authority = json.loads(evaluation_row["authority_json"])
            authority["normalized_result"]["decisions"][0][
                "reason_code"
            ] = "forged_authority_reason"
            authority["normalized_result_digest"] = canonical_digest(
                authority["normalized_result"]
            )
            payload = json.loads(event["payload_json"])
            payload["authority"] = authority
            metadata = json.loads(event["metadata_json"])
            envelope = {
                "event_id": event["event_id"],
                "stream_id": event["stream_id"],
                "stream_version": event["stream_version"],
                "event_type": event["event_type"],
                "schema_version": event["schema_version"],
                "occurred_at": event["occurred_at"],
                "recorded_at": event["recorded_at"],
                "learner_id": event["learner_id"],
                "session_id": event["session_id"],
                "correlation_id": event["correlation_id"],
                "causation_id": event["causation_id"],
                "idempotency_key": event["idempotency_key"],
                "payload": payload,
                "metadata": metadata,
                "previous_hash": event["previous_hash"],
            }
            connection.execute(
                """UPDATE task_evaluations SET authority_json=? WHERE id=?""",
                (canonical_json(authority), evaluation_row["id"]),
            )
            connection.execute(
                """UPDATE events SET payload_json=?, payload_hash=?
                   WHERE event_id=?""",
                (
                    canonical_json(payload),
                    hashlib.sha256(
                        canonical_json(envelope).encode("utf-8")
                    ).hexdigest(),
                    event["event_id"],
                ),
            )

        self._assert_profile_fails_closed(
            "normalized authority is invalid"
        )

    def test_report_rejects_evaluation_action_boundary_drift(self) -> None:
        attempt_id = self._complete_attempt()
        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER task_evaluations_no_update")
            connection.execute(
                """UPDATE task_evaluations SET through_sequence=0
                   WHERE attempt_id=?""",
                (attempt_id,),
            )

        self._assert_profile_fails_closed("submitted action boundary")

    def test_report_rejects_missing_or_corrupt_shadow_bundle(self) -> None:
        attempt_id = self._complete_attempt()
        with self.database.transaction() as connection:
            connection.execute(
                "DROP TRIGGER shadow_evidence_bundles_no_delete"
            )
            connection.execute(
                "DELETE FROM shadow_evidence_bundles WHERE attempt_id=?",
                (attempt_id,),
            )

        self._assert_profile_fails_closed("lacks an attempt-matched shadow bundle")

    def test_report_recomputes_and_rejects_corrupt_shadow_metrics(self) -> None:
        attempt_id = self._complete_attempt()
        with self.database.transaction() as connection:
            connection.execute(
                "DROP TRIGGER shadow_evidence_bundles_no_update"
            )
            connection.execute(
                """UPDATE shadow_evidence_bundles
                   SET bundle_json=json_set(
                       bundle_json, '$.reported_task_score', 0.99
                   )
                   WHERE attempt_id=?""",
                (attempt_id,),
            )

        self._assert_profile_fails_closed("deterministic reduction")


if __name__ == "__main__":
    unittest.main()
