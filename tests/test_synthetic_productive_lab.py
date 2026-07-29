# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.errors import NotFoundError, ValidationError
from tsq.evidence import (
    ActionKind,
    ActionPhase,
    CriterionScale,
    LearningAction,
    LearningTask,
    RubricCriterion,
    TaskModality,
    canonical_digest,
)
from tsq.performance_ledger import (
    PerformanceLedger,
    PerformanceTaskRelease,
    SyntheticTaskLabDeclaration,
    TaskReleaseReview,
    derive_performance_projections,
    read_task_release,
    rebuild_performance_projections,
)
from tsq.performance_selection import (
    inspect_synthetic_lab_tasks,
    recommend_performance_tasks,
)
from tsq.store import Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"
START = datetime(2118, 4, 5, 9, 0, tzinfo=timezone.utc)
_D0 = "0" * 64
_D1 = "1" * 64
_D2 = "2" * 64
_D3 = "3" * 64


class SyntheticProductiveLabTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.database = Database(
            Path(self.tempdir.name) / "synthetic-productive-lab.db"
        )
        self.database.initialize()
        self.database.import_corpus(
            *read_and_parse(CORPUS, include_catalog=True)
        )
        self.engine = AdaptiveEngine(self.database)
        self.learner_id = "synthetic-productive-lab-learner"
        self.engine.create_learner(self.learner_id)
        self.session = self.engine.start_session(
            self.learner_id,
            "t_transformers",
            seed=8118,
            now=START,
        )
        presentation = self.engine.next_question(
            self.session["id"],
            now=START + timedelta(minutes=1),
        )
        selected = sorted(
            (
                option
                for option in presentation.question.options
                if not option.correct
            ),
            key=lambda option: (
                option.misconception_id is None,
                option.id,
            ),
        )[0]
        self.engine.submit_answer(
            presentation.decision_id,
            selected.id,
            confidence=0.8,
            response_ms=8_000,
            hint_count=0,
            idempotency_key="synthetic-productive-lab-wrong-answer",
            now=START + timedelta(minutes=2),
        )
        focused = self.database.get_session(self.session["id"])
        self.corpus_release_id = focused["corpus_release_id"]
        self.objective_id = (
            focused["focus_objective_id"]
            or presentation.question.objective_id
        )
        if self.objective_id is None:
            raise AssertionError(
                "Synthetic productive lab fixture needs an objective-bound "
                "selected-response question."
            )
        objectives = {
            objective.id: objective
            for objective in self.database.get_learning_objectives(
                self.corpus_release_id
            )
        }
        self.concept_id = objectives[
            self.objective_id
        ].primary_concept_id
        source_id = sorted(presentation.question.source_ids)[0]
        with self.database.read() as connection:
            source = connection.execute(
                """SELECT source.id, source.content_hash
                   FROM release_sources membership
                   JOIN sources source
                     ON source.id=membership.source_id
                   WHERE membership.release_id=?
                     AND membership.source_id=?""",
                (self.corpus_release_id, source_id),
            ).fetchone()
        if source is None:
            raise AssertionError(
                "Synthetic productive lab fixture lost its pinned source."
            )
        self.source_manifests = (
            (source["id"], source["content_hash"]),
        )
        self.declaration = SyntheticTaskLabDeclaration(
            producer_id="synthetic.productive-lab-tests",
            producer_version="v1",
            declared_at=START.isoformat(),
            manifest_digest=_D3,
        )
        self.exact_task = self._task(
            "task_synthetic_lab_exact",
            "family_synthetic_lab_used",
            _D0,
        )
        self.same_family_task = self._task(
            "task_synthetic_lab_same_family",
            "family_synthetic_lab_used",
            _D1,
        )
        self.fresh_family_task = self._task(
            "task_synthetic_lab_fresh_family",
            "family_synthetic_lab_fresh",
            _D2,
        )
        self.ledger = PerformanceLedger(self.database)

    def _task(
        self,
        task_id: str,
        family_id: str,
        stimulus_digest: str,
    ) -> LearningTask:
        return LearningTask(
            id=task_id,
            version=1,
            family_id=family_id,
            title=f"Synthetic laboratory task {task_id}",
            modality=TaskModality.DEBUGGING,
            criteria=(
                RubricCriterion(
                    id=f"criterion_{task_id}",
                    name="Synthetic objective diagnosis",
                    scale=CriterionScale.CONTINUOUS,
                    concept_weights=((self.concept_id, 1.0),),
                    objective_weights=((self.objective_id, 1.0),),
                    dependence_group=f"dependence_{task_id}",
                    evidence_cap=0.0,
                    dependence_cap=0.0,
                    assisted_evidence_factor=0.0,
                    certification_eligible=False,
                ),
            ),
            instructions=(
                "Inspect the pinned synthetic laboratory stimulus and return "
                "a content-addressed diagnosis."
            ),
            source_manifests=self.source_manifests,
            administration_id="synthetic_productive_lab_admin_v1",
            administration_manifest_digest=_D0,
            stimulus_id=f"stimulus_{task_id}",
            stimulus_digest=stimulus_digest,
            scorer_contracts=(),
            evidence_cap=0.0,
        )

    def _bundle(
        self,
        *tasks: LearningTask,
        title: str = "Synthetic productive laboratory release",
    ) -> PerformanceTaskRelease:
        return PerformanceTaskRelease(
            title=title,
            corpus_release_id=self.corpus_release_id,
            review=self.declaration,
            tasks=tuple(("quarantined", task) for task in tasks),
            schema_version=2,
        )

    def _event_count(self) -> int:
        with self.database.read() as connection:
            return connection.execute(
                "SELECT COUNT(*) AS n FROM events"
            ).fetchone()["n"]

    def _publish_human_pilot(self) -> dict[str, object]:
        bundle = PerformanceTaskRelease(
            title="Human replay boundary fixture",
            corpus_release_id=self.corpus_release_id,
            review=TaskReleaseReview(
                reviewer_kind="human",
                reviewer_id="independent_replay_boundary_reviewer",
                reviewed_at=START.isoformat(),
                independent_of_author=True,
                attestation_digest=_D2,
            ),
            tasks=(("pilot", self.exact_task),),
        )
        return self.ledger.publish_release(
            bundle,
            now=START + timedelta(minutes=3),
        )

    def _append_start_event(
        self,
        release_id: str,
        *,
        learner_id: str | None = None,
        session_revision_delta: int = 0,
        learner_revision_delta: int = 0,
        command_hash: str | None = None,
    ) -> None:
        event_learner_id = learner_id or self.learner_id
        with self.database.transaction() as connection:
            session = connection.execute(
                "SELECT * FROM sessions WHERE id=?",
                (self.session["id"],),
            ).fetchone()
            learner = connection.execute(
                "SELECT * FROM learners WHERE id=?",
                (event_learner_id,),
            ).fetchone()
            if command_hash is None:
                command_hash = canonical_digest(
                    {
                        "type": "tsq.performance_command",
                        "operation": "start_attempt",
                        "session_id": self.session["id"],
                        "task_id": self.exact_task.id,
                        "task_version": self.exact_task.version,
                        "task_release_id": release_id,
                    }
                )
            self.database.append_event(
                connection,
                stream_id=f"learner:{event_learner_id}",
                event_type="PerformanceTaskStarted",
                schema_version=1,
                payload={
                    "attempt_id": "pta_forged_human_start",
                    "session_id": self.session["id"],
                    "learner_id": event_learner_id,
                    "task_release_id": release_id,
                    "corpus_release_id": self.corpus_release_id,
                    "task_id": self.exact_task.id,
                    "task_version": self.exact_task.version,
                    "task_digest": self.exact_task.digest,
                    "session_revision": (
                        session["revision"] + session_revision_delta
                    ),
                    "learner_revision": (
                        learner["revision"] + learner_revision_delta
                    ),
                },
                metadata={
                    "command_hash": command_hash,
                    "task_schema_version": self.exact_task.schema_version,
                    "shadow_only": True,
                    "projection_applied": False,
                    "certification_applied": False,
                },
                learner_id=event_learner_id,
                session_id=self.session["id"],
                correlation_id="pta_forged_human_start",
                causation_id=self.session["id"],
                occurred_at=START + timedelta(minutes=4),
            )

    def _append_action_event(
        self,
        attempt_id: str,
        *,
        kind: ActionKind,
        payload: dict[str, object],
        command_hash: str | None = None,
    ) -> None:
        with self.database.transaction() as connection:
            attempt = connection.execute(
                "SELECT * FROM performance_attempts WHERE id=?",
                (attempt_id,),
            ).fetchone()
            action = LearningAction(
                id=f"pact_forged_{kind.value}",
                trace_id=attempt_id,
                sequence=1,
                kind=kind,
                phase=ActionPhase.UNASSISTED,
                payload=payload,
                elapsed_ms=60_000,
            )
            if command_hash is None:
                command_hash = canonical_digest(
                    {
                        "type": "tsq.performance_command",
                        "operation": "record_action",
                        "attempt_id": attempt_id,
                        "action_type": kind.value,
                        "phase": ActionPhase.UNASSISTED.value,
                        "payload": payload,
                    }
                )
            self.database.append_event(
                connection,
                stream_id=f"learner:{attempt['learner_id']}",
                event_type="PerformanceActionRecorded",
                schema_version=1,
                payload={
                    "attempt_id": attempt_id,
                    "action": action.terms(),
                },
                metadata={
                    "command_hash": command_hash,
                    "action_schema_version": action.schema_version,
                    "task_digest": attempt["task_digest"],
                    "task_release_id": attempt["task_release_id"],
                    "corpus_release_id": attempt["corpus_release_id"],
                    "observational_only": True,
                    "shadow_only": True,
                },
                learner_id=attempt["learner_id"],
                session_id=attempt["session_id"],
                correlation_id=attempt_id,
                causation_id=attempt_id,
                occurred_at=START + timedelta(minutes=5),
            )

    def test_declaration_terms_round_trip_through_strict_release_parser(
        self,
    ) -> None:
        bundle = self._bundle(self.exact_task)
        self.assertEqual(
            self.declaration.terms(),
            {
                "declaration_kind": "synthetic_lab",
                "producer_id": "synthetic.productive-lab-tests",
                "producer_version": "v1",
                "declared_at": START.isoformat(),
                "manifest_digest": _D3,
                "human_reviewed": False,
                "activation_authority": False,
            },
        )
        decoded = PerformanceTaskRelease.from_terms(bundle.terms())
        self.assertEqual(decoded.terms(), bundle.terms())
        self.assertEqual(decoded.bundle_hash, bundle.bundle_hash)
        self.assertIsInstance(
            decoded.review,
            SyntheticTaskLabDeclaration,
        )

        for field in ("producer_version", "manifest_digest"):
            with self.subTest(missing=field):
                malformed = deepcopy(bundle.terms())
                del malformed["review"][field]
                with self.assertRaises(ValidationError):
                    PerformanceTaskRelease.from_terms(malformed)
        malformed = deepcopy(bundle.terms())
        malformed["review"]["unexpected"] = True
        with self.assertRaises(ValidationError):
            PerformanceTaskRelease.from_terms(malformed)
        for invalid_schema_version in (True, 2.0, 1, 3):
            with self.subTest(
                invalid_schema_version=invalid_schema_version
            ):
                malformed = deepcopy(bundle.terms())
                malformed["schema_version"] = invalid_schema_version
                with self.assertRaises(ValidationError):
                    PerformanceTaskRelease.from_terms(malformed)

    def test_human_v1_fixture_identity_is_unchanged(self) -> None:
        release = read_task_release(
            ROOT / "tests" / "fixtures" / "reviewed_productive_task_release.json"
        )

        self.assertIsInstance(release.review, TaskReleaseReview)
        self.assertEqual(release.schema_version, 1)
        self.assertEqual(
            release.bundle_hash,
            "66945c583dcfa564940a46344d2487aa5411ad4835d3be28c5050922f3f87034",
        )
        self.assertEqual(
            release.release_id,
            "ptrel_66945c583dcfa564940a4634",
        )

    def test_synthetic_release_requires_quarantine_and_zero_authority(
        self,
    ) -> None:
        for status in ("pilot", "approved"):
            with self.subTest(status=status):
                with self.assertRaisesRegex(
                    ValidationError,
                    "quarantined",
                ):
                    PerformanceTaskRelease(
                        title="Synthetic serviceability rejection",
                        corpus_release_id=self.corpus_release_id,
                        review=self.declaration,
                        tasks=((status, self.exact_task),),
                        schema_version=2,
                    )

        criterion = replace(
            self.exact_task.criteria[0],
            evidence_cap=0.1,
            dependence_cap=0.1,
            certification_eligible=True,
        )
        authority_bearing = replace(
            self.exact_task,
            criteria=(criterion,),
            evidence_cap=0.1,
        )
        with self.assertRaisesRegex(
            ValidationError,
            "evidence|certification|synthetic",
        ):
            self._bundle(authority_bearing)

    def test_explicit_publish_is_idempotent_listed_and_integral(
        self,
    ) -> None:
        bundle = self._bundle(
            self.exact_task,
            self.fresh_family_task,
        )
        with self.database.read() as connection:
            before_releases = connection.execute(
                "SELECT COUNT(*) AS n FROM performance_task_releases"
            ).fetchone()["n"]
        with self.assertRaisesRegex(ValidationError, "synthetic"):
            self.ledger.publish_release(
                bundle,
                now=START + timedelta(minutes=3),
            )
        with self.database.read() as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) AS n FROM performance_task_releases"
                ).fetchone()["n"],
                before_releases,
            )

        first = self.ledger.publish_synthetic_lab_release(
            bundle,
            now=START + timedelta(minutes=3),
        )
        second = self.ledger.publish_synthetic_lab_release(
            bundle,
            now=START + timedelta(minutes=4),
        )
        self.assertFalse(first["idempotent_replay"])
        self.assertTrue(second["idempotent_replay"])
        self.assertEqual(first["release_authority_kind"], "synthetic_lab")
        self.assertEqual(second["release_authority_kind"], "synthetic_lab")
        self.assertEqual(first["release_id"], bundle.release_id)
        self.assertEqual(second["release_id"], first["release_id"])
        self.assertEqual(
            first["status_counts"],
            {
                "approved": 0,
                "pilot": 0,
                "quarantined": 2,
            },
        )

        listed = next(
            release
            for release in self.ledger.list_releases()
            if release["id"] == first["release_id"]
        )
        self.assertEqual(
            listed["review"],
            self.declaration.terms(),
        )
        self.assertEqual(listed["approved_count"], 0)
        self.assertEqual(listed["pilot_count"], 0)
        self.assertEqual(listed["quarantined_count"], 2)
        listed_tasks = self.ledger.list_tasks(
            release_id=first["release_id"]
        )
        self.assertTrue(listed_tasks)
        self.assertTrue(
            all(
                task["release_authority_kind"] == "synthetic_lab"
                for task in listed_tasks
            )
        )
        shown = self.ledger.show_task(
            self.exact_task.id,
            task_version=self.exact_task.version,
            release_id=first["release_id"],
        )
        self.assertEqual(
            shown["release_authority_kind"], "synthetic_lab"
        )
        integrity = self.database.verify_integrity()
        self.assertTrue(integrity["ok"], integrity["errors"])

    def test_idempotent_publish_reconstructs_the_complete_stored_bundle(
        self,
    ) -> None:
        bundle = self._bundle(self.exact_task)
        timestamp = (START + timedelta(minutes=3)).isoformat()
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO performance_task_releases(
                       id, corpus_release_id, bundle_hash, title,
                       review_json, created_at, sealed_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    bundle.release_id,
                    bundle.corpus_release_id,
                    bundle.bundle_hash,
                    "Forged partial replay row",
                    json.dumps(
                        bundle.review.terms(),
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    timestamp,
                    timestamp,
                ),
            )

        with self.assertRaisesRegex(
            ValidationError,
            "reconstruct|commitment|tasks",
        ):
            self.ledger.publish_synthetic_lab_release(
                bundle,
                now=START + timedelta(minutes=4),
            )

    def test_existing_task_cannot_be_backdated_into_a_new_release(self) -> None:
        later = self._bundle(
            self.exact_task,
            title="Later synthetic task import",
        )
        self.ledger.publish_synthetic_lab_release(
            later,
            now=START + timedelta(minutes=10),
        )
        earlier = self._bundle(
            self.exact_task,
            title="Earlier synthetic reuse",
        )

        with self.assertRaisesRegex(
            ValidationError,
            "predates its immutable import",
        ):
            self.ledger.publish_synthetic_lab_release(
                earlier,
                now=START + timedelta(minutes=5),
            )
        integrity = self.database.verify_integrity()
        self.assertTrue(integrity["ok"], integrity["errors"])

    def test_synthetic_release_is_inert_to_normal_runtime_paths(
        self,
    ) -> None:
        bundle = self._bundle(self.exact_task)
        released = self.ledger.publish_synthetic_lab_release(
            bundle,
            now=START + timedelta(minutes=3),
        )
        before_events = self._event_count()
        with self.assertRaisesRegex(
            NotFoundError,
            "No serviceable release",
        ):
            self.ledger.start_attempt(
                self.session["id"],
                self.exact_task.id,
                task_version=self.exact_task.version,
                task_release_id=released["release_id"],
                idempotency_key="synthetic-lab-normal-start-denied",
                now=START + timedelta(minutes=4),
            )
        self.assertEqual(self._event_count(), before_events)

        normal = recommend_performance_tasks(
            self.database,
            self.session["id"],
            limit=20,
            now=START + timedelta(minutes=5),
        )
        self.assertNotIn(
            self.exact_task.id,
            {
                item["task_id"]
                for item in normal["recommendations"]
            },
        )
        self.assertEqual(self._event_count(), before_events)

    def test_tampered_synthetic_authority_fails_integrity(self) -> None:
        bundle = self._bundle(self.exact_task)
        released = self.ledger.publish_synthetic_lab_release(
            bundle,
            now=START + timedelta(minutes=3),
        )
        forged = self.declaration.terms()
        forged["activation_authority"] = True
        with self.database.transaction() as connection:
            connection.execute(
                "DROP TRIGGER performance_task_releases_no_update"
            )
            connection.execute(
                """UPDATE performance_task_releases
                   SET review_json=? WHERE id=?""",
                (
                    json.dumps(
                        forged,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    released["release_id"],
                ),
            )

        integrity = self.database.verify_integrity()
        self.assertFalse(integrity["ok"])
        self.assertTrue(
            any(
                "activation authority" in error
                or "bundle commitment mismatch" in error
                for error in integrity["errors"]
            ),
            integrity["errors"],
        )

    def test_inspector_is_release_scoped_and_applies_prior_family_novelty(
        self,
    ) -> None:
        target = self._bundle(
            self.exact_task,
            self.same_family_task,
            self.fresh_family_task,
        )
        target_release = self.ledger.publish_synthetic_lab_release(
            target,
            now=START + timedelta(minutes=3),
        )
        other_task = self._task(
            "task_synthetic_lab_other_release",
            "family_synthetic_lab_other_release",
            _D3,
        )
        other = self._bundle(
            other_task,
            title="Other synthetic productive laboratory release",
        )
        self.ledger.publish_synthetic_lab_release(
            other,
            now=START + timedelta(minutes=4),
        )

        before_events = self._event_count()
        initial = inspect_synthetic_lab_tasks(
            self.database,
            self.session["id"],
            target_release["release_id"],
            limit=5,
            now=START + timedelta(minutes=5),
        )
        self.assertEqual(initial["eligible_candidate_count"], 3)
        self.assertEqual(initial["ranked_family_count"], 2)
        self.assertEqual(len(initial["recommendations"]), 2)
        self.assertEqual(
            {
                item["task_release_id"]
                for item in initial["recommendations"]
            },
            {target_release["release_id"]},
        )
        self.assertNotIn(
            other_task.id,
            {
                item["task_id"]
                for item in initial["recommendations"]
            },
        )

        after_prior = inspect_synthetic_lab_tasks(
            self.database,
            self.session["id"],
            target_release["release_id"],
            prior_task_refs=(
                (
                    self.exact_task.id,
                    self.exact_task.version,
                    self.exact_task.digest,
                ),
            ),
            limit=5,
            now=START + timedelta(minutes=5),
        )
        self.assertEqual(
            after_prior["recommendations"][0]["family_id"],
            self.fresh_family_task.family_id,
        )
        self.assertTrue(
            after_prior["fresh_family_constraint_applied"],
        )
        self.assertEqual(
            {
                item["family_id"]
                for item in after_prior["recommendations"]
            },
            {self.fresh_family_task.family_id},
        )
        self.assertTrue(
            all(
                item["prior_family_attempts"] == 0
                for item in after_prior["recommendations"]
            )
        )
        self.assertEqual(self._event_count(), before_events)

    def test_inspector_history_uses_repeatable_exact_task_references(
        self,
    ) -> None:
        version_two = replace(
            self.exact_task,
            version=2,
            stimulus_digest=_D1,
        )
        bundle = self._bundle(self.exact_task, version_two)
        released = self.ledger.publish_synthetic_lab_release(
            bundle,
            now=START + timedelta(minutes=3),
        )
        exact_ref = (
            self.exact_task.id,
            self.exact_task.version,
            self.exact_task.digest,
        )
        report = inspect_synthetic_lab_tasks(
            self.database,
            self.session["id"],
            released["release_id"],
            prior_task_refs=(exact_ref, exact_ref),
            limit=5,
            now=START + timedelta(minutes=4),
        )

        self.assertEqual(
            report["synthetic_prior_task_refs"],
            [
                {
                    "task_id": self.exact_task.id,
                    "task_version": self.exact_task.version,
                    "task_digest": self.exact_task.digest,
                },
                {
                    "task_id": self.exact_task.id,
                    "task_version": self.exact_task.version,
                    "task_digest": self.exact_task.digest,
                },
            ],
        )
        self.assertEqual(
            report["recommendations"][0]["prior_family_attempts"],
            2,
        )
        version_two_ref = (
            version_two.id,
            version_two.version,
            version_two.digest,
        )
        inspect_synthetic_lab_tasks(
            self.database,
            self.session["id"],
            released["release_id"],
            prior_task_refs=(version_two_ref,),
            now=START + timedelta(minutes=4),
        )
        with self.assertRaisesRegex(
            ValidationError,
            "outside the exact",
        ):
            inspect_synthetic_lab_tasks(
                self.database,
                self.session["id"],
                released["release_id"],
                prior_task_refs=(
                    (
                        version_two.id,
                        version_two.version,
                        self.exact_task.digest,
                    ),
                ),
                now=START + timedelta(minutes=4),
            )

    def test_inspector_rejects_future_or_partially_tampered_release(
        self,
    ) -> None:
        bundle = self._bundle(
            self.exact_task,
            self.fresh_family_task,
        )
        released = self.ledger.publish_synthetic_lab_release(
            bundle,
            now=START + timedelta(minutes=10),
        )
        with self.assertRaisesRegex(ValidationError, "before.*publication"):
            inspect_synthetic_lab_tasks(
                self.database,
                self.session["id"],
                released["release_id"],
                now=START + timedelta(minutes=5),
            )
        normal = recommend_performance_tasks(
            self.database,
            self.session["id"],
            now=START + timedelta(minutes=5),
        )
        self.assertEqual(normal["recommendations"], [])

        with self.database.transaction() as connection:
            connection.execute(
                "DROP TRIGGER release_performance_tasks_no_update"
            )
            connection.execute(
                """UPDATE release_performance_tasks SET status='pilot'
                   WHERE release_id=? AND task_id=? AND task_version=?""",
                (
                    released["release_id"],
                    self.fresh_family_task.id,
                    self.fresh_family_task.version,
                ),
            )
        with self.assertRaisesRegex(
            ValidationError,
            "quarantined|commitment",
        ):
            inspect_synthetic_lab_tasks(
                self.database,
                self.session["id"],
                released["release_id"],
                now=START + timedelta(minutes=11),
            )

    def test_synthetic_start_event_cannot_enter_derived_projections(
        self,
    ) -> None:
        bundle = self._bundle(self.exact_task)
        released = self.ledger.publish_synthetic_lab_release(
            bundle,
            now=START + timedelta(minutes=3),
        )
        with self.database.transaction() as connection:
            session = connection.execute(
                "SELECT * FROM sessions WHERE id=?",
                (self.session["id"],),
            ).fetchone()
            learner = connection.execute(
                "SELECT * FROM learners WHERE id=?",
                (self.learner_id,),
            ).fetchone()
            self.database.append_event(
                connection,
                stream_id=f"learner:{self.learner_id}",
                event_type="PerformanceTaskStarted",
                schema_version=1,
                payload={
                    "attempt_id": "pta_forged_synthetic_start",
                    "session_id": self.session["id"],
                    "learner_id": self.learner_id,
                    "task_release_id": released["release_id"],
                    "corpus_release_id": self.corpus_release_id,
                    "task_id": self.exact_task.id,
                    "task_version": self.exact_task.version,
                    "task_digest": self.exact_task.digest,
                    "session_revision": session["revision"],
                    "learner_revision": learner["revision"],
                },
                metadata={
                    "command_hash": _D0,
                    "task_schema_version": self.exact_task.schema_version,
                    "shadow_only": True,
                    "projection_applied": False,
                    "certification_applied": False,
                },
                learner_id=self.learner_id,
                session_id=self.session["id"],
                correlation_id="pta_forged_synthetic_start",
                causation_id=self.session["id"],
                occurred_at=START + timedelta(minutes=4),
            )

        with self.database.read() as connection:
            with self.assertRaisesRegex(
                ValidationError,
                "human-reviewed",
            ):
                derive_performance_projections(connection)
        with self.database.transaction() as connection:
            with self.assertRaisesRegex(
                ValidationError,
                "human-reviewed",
            ):
                rebuild_performance_projections(connection)
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) AS n FROM performance_attempts"
                ).fetchone()["n"],
                0,
            )

    def test_synthetic_start_event_cannot_enter_learner_reports(self) -> None:
        released = self.ledger.publish_synthetic_lab_release(
            self._bundle(self.exact_task),
            now=START + timedelta(minutes=3),
        )
        self._append_start_event(str(released["release_id"]))

        for operation in (
            lambda: self.engine.profile(
                self.learner_id,
                now=START + timedelta(minutes=5),
            ),
            lambda: self.engine.session_report(
                self.session["id"],
                now=START + timedelta(minutes=5),
            ),
            lambda: recommend_performance_tasks(
                self.database,
                self.session["id"],
                now=START + timedelta(minutes=5),
            ),
        ):
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(
                    ValidationError,
                    "human-reviewed",
                ):
                    operation()

    def test_derived_start_rejects_forged_revision_snapshot(self) -> None:
        released = self._publish_human_pilot()
        self._append_start_event(
            str(released["release_id"]),
            session_revision_delta=7,
            learner_revision_delta=7,
        )

        with self.database.read() as connection:
            with self.assertRaisesRegex(
                ValidationError,
                "revision snapshot",
            ):
                derive_performance_projections(connection)

    def test_derived_start_rejects_forged_command_commitment(self) -> None:
        released = self._publish_human_pilot()
        self._append_start_event(
            str(released["release_id"]),
            command_hash="f" * 64,
        )

        with self.database.read() as connection:
            with self.assertRaisesRegex(
                ValidationError,
                "command commitment",
            ):
                derive_performance_projections(connection)

    def test_derived_start_rejects_cross_learner_session(self) -> None:
        released = self._publish_human_pilot()
        other_learner_id = "synthetic-replay-other-learner"
        self.engine.create_learner(other_learner_id)
        self._append_start_event(
            str(released["release_id"]),
            learner_id=other_learner_id,
        )

        with self.database.read() as connection:
            with self.assertRaisesRegex(
                ValidationError,
                "ownership|corpus",
            ):
                derive_performance_projections(connection)

    def test_derived_start_rejects_pending_selected_response(self) -> None:
        released = self._publish_human_pilot()
        self.engine.next_question(
            self.session["id"],
            now=START + timedelta(minutes=3, seconds=30),
        )
        self._append_start_event(str(released["release_id"]))

        with self.database.read() as connection:
            with self.assertRaisesRegex(
                ValidationError,
                "remain pending",
            ):
                derive_performance_projections(connection)

    def test_derived_start_rejects_overlapping_open_attempt(self) -> None:
        released = self._publish_human_pilot()
        self.ledger.start_attempt(
            self.session["id"],
            self.exact_task.id,
            task_version=self.exact_task.version,
            task_release_id=str(released["release_id"]),
            now=START + timedelta(minutes=4),
        )
        self._append_start_event(str(released["release_id"]))

        with self.database.read() as connection:
            with self.assertRaisesRegex(
                ValidationError,
                "overlaps active attempt",
            ):
                derive_performance_projections(connection)

    def test_derived_action_rejects_forged_command_commitment(self) -> None:
        released = self._publish_human_pilot()
        attempt = self.ledger.start_attempt(
            self.session["id"],
            self.exact_task.id,
            task_version=self.exact_task.version,
            task_release_id=str(released["release_id"]),
            now=START + timedelta(minutes=4),
        )
        self._append_action_event(
            attempt["id"],
            kind=ActionKind.ABANDONED,
            payload={"reason_code": "synthetic_redteam"},
            command_hash="f" * 64,
        )

        with self.database.read() as connection:
            with self.assertRaisesRegex(
                ValidationError,
                "action command commitment",
            ):
                derive_performance_projections(connection)

    def test_derived_action_obeys_exact_task_allowlist(self) -> None:
        restricted_task = replace(
            self.exact_task,
            id="task_synthetic_lab_restricted_actions",
            allowed_action_kinds=(
                ActionKind.STARTED,
                ActionKind.ABANDONED,
            ),
        )
        bundle = PerformanceTaskRelease(
            title="Restricted action replay boundary",
            corpus_release_id=self.corpus_release_id,
            review=TaskReleaseReview(
                reviewer_kind="human",
                reviewer_id="independent_replay_boundary_reviewer",
                reviewed_at=START.isoformat(),
                independent_of_author=True,
                attestation_digest=_D2,
            ),
            tasks=(("pilot", restricted_task),),
        )
        released = self.ledger.publish_release(
            bundle,
            now=START + timedelta(minutes=3),
        )
        attempt = self.ledger.start_attempt(
            self.session["id"],
            restricted_task.id,
            task_version=restricted_task.version,
            task_release_id=str(released["release_id"]),
            now=START + timedelta(minutes=4),
        )
        self._append_action_event(
            attempt["id"],
            kind=ActionKind.HINT_REQUESTED,
            payload={"hint_id": "hint_cache_key", "level": 1},
        )

        with self.database.read() as connection:
            with self.assertRaisesRegex(
                ValidationError,
                "not allowed by the exact task release",
            ):
                derive_performance_projections(connection)

    def test_inspector_rejects_a_human_reviewed_release(self) -> None:
        human_bundle = PerformanceTaskRelease(
            title="Human-reviewed quarantine fixture",
            corpus_release_id=self.corpus_release_id,
            review=TaskReleaseReview(
                reviewer_kind="human",
                reviewer_id="independent_synthetic_boundary_reviewer",
                reviewed_at=START.isoformat(),
                independent_of_author=True,
                attestation_digest=_D2,
            ),
            tasks=(("quarantined", self.exact_task),),
        )
        with self.assertRaisesRegex(
            ValidationError,
            "SyntheticTaskLabDeclaration",
        ):
            self.ledger.publish_synthetic_lab_release(
                human_bundle,
                now=START + timedelta(minutes=3),
            )
        released = self.ledger.publish_release(
            human_bundle,
            now=START + timedelta(minutes=3),
        )
        before_events = self._event_count()
        with self.assertRaisesRegex(
            ValidationError,
            "synthetic",
        ):
            inspect_synthetic_lab_tasks(
                self.database,
                self.session["id"],
                released["release_id"],
                now=START + timedelta(minutes=4),
            )
        self.assertEqual(self._event_count(), before_events)
        for invalid_release_id in (None, "", 7):
            with self.subTest(invalid_release_id=invalid_release_id):
                with self.assertRaises(ValidationError):
                    inspect_synthetic_lab_tasks(
                        self.database,
                        self.session["id"],
                        invalid_release_id,  # type: ignore[arg-type]
                        now=START + timedelta(minutes=4),
                    )


if __name__ == "__main__":
    unittest.main()
