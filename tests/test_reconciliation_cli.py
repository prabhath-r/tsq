# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from tsq.cli import main
from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.errors import ValidationError
from tsq.evidence import ActionPhase, EvaluationStatus, ScorerKind
from tsq.performance import (
    ImportedCriterionResult,
    ImportedEvaluation,
    ProviderAuthorityBinding,
    ScoringProviderRegistry,
    SyntheticDeterministicProvider,
)
from tsq.performance_ledger import (
    PerformanceLedger,
    read_task_release,
)
from tsq.store import Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
TASK_RELEASE_FIXTURE = (
    ROOT / "tests" / "fixtures" / "reviewed_productive_task_release.json"
)
_CORPUS_PLACEHOLDER = "rel_fixture_requires_explicit_pinning"
_D3 = "3" * 64


class ReconciliationCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.database = Database(Path(self.tempdir.name) / "reconciliation.db")
        self.database.initialize()
        self.database.import_corpus(
            *read_and_parse(CORPUS, include_catalog=True)
        )
        self.engine = AdaptiveEngine(self.database)
        self.engine.create_learner(
            "reconciliation-cli-learner",
            "Reconciliation CLI Learner",
        )
        self.session = self.engine.start_session(
            "reconciliation-cli-learner",
            "t_transformers",
            seed=2718,
        )
        with self.database.read() as connection:
            corpus_release_id = connection.execute(
                "SELECT value FROM meta WHERE key='active_corpus_release'"
            ).fetchone()["value"]
        template = read_task_release(TASK_RELEASE_FIXTURE)
        self.assertEqual(template.corpus_release_id, _CORPUS_PLACEHOLDER)
        release = replace(template, corpus_release_id=corpus_release_id)
        self.ledger = PerformanceLedger(self.database)
        release_report = self.ledger.publish_release(release)
        self.task = release.tasks[0][1]
        attempt = self.ledger.start_attempt(
            self.session["id"],
            self.task.id,
            task_version=self.task.version,
            task_release_id=release_report["release_id"],
            idempotency_key="reconciliation-cli-attempt",
        )
        self.attempt_id = attempt["id"]
        self.ledger.record_action(
            self.attempt_id,
            "artifact_checkpoint",
            {
                "artifact_digest": _D3,
                "artifact_kind": "synthetic_cli_patch",
            },
            idempotency_key="reconciliation-cli-artifact",
        )
        submitted = self.ledger.record_action(
            self.attempt_id,
            "submitted",
            {"submission_digest": _D3},
            phase=ActionPhase.ASSISTED.value,
            idempotency_key="reconciliation-cli-submit",
        )
        imported = ImportedEvaluation(
            criteria=tuple(
                ImportedCriterionResult(
                    criterion_id=criterion.id,
                    status=EvaluationStatus.INVALID,
                    score=None,
                    outcome_code="synthetic_callback_interrupted",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(submitted["id"],),
                    reliability=1.0,
                )
                for criterion in self.task.criteria
            )
        )

        class FailingProvider(SyntheticDeterministicProvider):
            def __init__(self, evaluation):
                super().__init__(evaluation)
                self.calls = 0

            def score(self, request):
                self.calls += 1
                raise RuntimeError("synthetic callback interruption")

        self.failing_provider = FailingProvider(imported)
        registry = ScoringProviderRegistry(allow_synthetic=True)
        registry.register(
            self.failing_provider,
            self.failing_provider.authority_binding,
        )
        self.private_scoring_key = "private-score-key-never-render"
        with self.assertRaisesRegex(ValidationError, "failed safely"):
            self.ledger.score_attempt(
                self.attempt_id,
                registry,
                self.failing_provider.provider_id,
                self.failing_provider.provider_version,
                idempotency_key=self.private_scoring_key,
            )
        claims = self.ledger.list_scoring_claims(
            attempt_id=self.attempt_id,
            status="unreconciled",
        )
        self.assertEqual(len(claims), 1)
        self.claim_id = claims[0]["id"]
        self.before_projection = self._projection_snapshot()

    def _projection_snapshot(self) -> tuple[int, int, str]:
        with self.database.read() as connection:
            learner_revision = connection.execute(
                """SELECT revision FROM learners
                   WHERE id='reconciliation-cli-learner'"""
            ).fetchone()["revision"]
            session_revision = connection.execute(
                "SELECT revision FROM sessions WHERE id=?",
                (self.session["id"],),
            ).fetchone()["revision"]
            projection_hash = self.database.learner_projection_hash(
                "reconciliation-cli-learner",
                connection,
            )
        return (learner_revision, session_revision, projection_hash)

    def run_cli(self, *arguments: str) -> tuple[int, str, str]:
        output = io.StringIO()
        error = io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            exit_code = main(
                [
                    "--db",
                    str(self.database.path),
                    *arguments,
                ]
            )
        return (exit_code, output.getvalue(), error.getvalue())

    def reconcile_json(
        self,
        outcome: str,
        *,
        key: str,
        score: str | None = None,
        reliability: str | None = None,
        claim_id: str | None = None,
    ) -> tuple[int, dict, str, str]:
        arguments = [
            "task",
            "reconcile-claim",
            self.claim_id if claim_id is None else claim_id,
            "--provider",
            "deterministic-test",
            "--test-outcome",
            outcome,
            "--idempotency-key",
            key,
            "--json",
        ]
        if score is not None:
            arguments.extend(("--score", score))
        if reliability is not None:
            arguments.extend(("--reliability", reliability))
        code, output, error = self.run_cli(*arguments)
        parsed = json.loads(output) if output else {}
        return (code, parsed, output, error)

    def assert_boundaries(self, result: dict) -> None:
        self.assertFalse(result["provider_callback_invoked"])
        self.assertFalse(result["automatic_retry_allowed"])
        self.assertFalse(result["projection_applied"])
        self.assertFalse(result["certification_applied"])
        self.assertFalse(result["mastery_applied"])
        self.assertFalse(result["skill_authority"])
        self.assertTrue(result["synthetic_test_provider"])

    def test_claim_inspection_is_read_only_explicit_and_redacted(self) -> None:
        code, output, error = self.run_cli(
            "task", "claim", self.claim_id, "--json"
        )
        self.assertEqual(code, 0, error)
        result = json.loads(output)
        self.assertEqual(result["claim"]["status"], "unreconciled")
        self.assertFalse(result["provider_callback_invoked"])
        self.assertFalse(result["automatic_retry_allowed"])
        self.assertFalse(result["projection_applied"])
        self.assertFalse(result["certification_applied"])
        self.assertFalse(result["mastery_applied"])
        self.assertNotIn(self.private_scoring_key, output)
        self.assertNotIn(str(self.database.path), output)
        self.assertNotIn("idempotency_key", result["claim"])
        self.assertEqual(self.failing_provider.calls, 1)

        code, plain, error = self.run_cli("task", "claim", self.claim_id)
        self.assertEqual(code, 0, error)
        self.assertIn("Provider scoring callback invoked: no", plain)
        self.assertIn("Automatic retry allowed: no", plain)
        self.assertIn("Mastery applied: no", plain)
        self.assertNotIn(self.private_scoring_key, plain)
        self.assertEqual(self._projection_snapshot(), self.before_projection)

    def test_unknown_outcome_remains_nonterminal_and_legacy_filter_matches(
        self,
    ) -> None:
        code, result, raw, error = self.reconcile_json(
            "unknown", key="reconcile-cli-unknown"
        )
        self.assertEqual(code, 0, error)
        self.assert_boundaries(result)
        self.assertEqual(result["claim"]["status"], "unknown")
        self.assertFalse(result["claim"]["terminal"])
        self.assertEqual(result["claim"]["reconciliation_count"], 1)
        self.assertEqual(self.failing_provider.calls, 1)
        self.assertNotIn(self.private_scoring_key, raw)
        self.assertNotIn(str(self.database.path), raw)

        for status in ("unknown", "unresolved"):
            code, output, error = self.run_cli(
                "task", "claims", "--status", status, "--json"
            )
            self.assertEqual(code, 0, error)
            listed = json.loads(output)
            self.assertEqual([item["id"] for item in listed], [self.claim_id])
        self.assertEqual(
            self.ledger.report(self.attempt_id)["evaluation_count"],
            0,
        )
        self.assertEqual(self._projection_snapshot(), self.before_projection)

    def test_definitely_absent_is_terminal_without_retry_or_evaluation(
        self,
    ) -> None:
        code, result, _, error = self.reconcile_json(
            "definitely_absent",
            key="reconcile-cli-absent",
        )
        self.assertEqual(code, 0, error)
        self.assert_boundaries(result)
        self.assertEqual(result["claim"]["status"], "definitely_absent")
        self.assertTrue(result["claim"]["terminal"])
        self.assertFalse(result["claim"]["automatic_retry_allowed"])
        self.assertEqual(self.failing_provider.calls, 1)
        self.assertEqual(
            self.ledger.report(self.attempt_id)["evaluation_count"],
            0,
        )
        self.assertEqual(self._projection_snapshot(), self.before_projection)
        code, plain, error = self.run_cli(
            "task",
            "reconcile-claim",
            self.claim_id,
            "--provider",
            "deterministic-test",
            "--test-outcome",
            "definitely_absent",
            "--idempotency-key",
            "reconcile-cli-absent",
        )
        self.assertEqual(code, 0, error)
        self.assertIn("Synthetic test reconciliation", plain)
        self.assertIn("Provider scoring callback invoked: no", plain)
        self.assertIn("Automatic retry allowed: no", plain)
        self.assertIn("Mastery applied: no", plain)
        self.assertNotIn(self.private_scoring_key, plain)

    def test_completed_is_shadow_only_and_projection_neutral(self) -> None:
        code, result, raw, error = self.reconcile_json(
            "completed",
            key="reconcile-cli-completed",
            score="0.8",
            reliability="0.9",
        )
        self.assertEqual(code, 0, error)
        self.assert_boundaries(result)
        self.assertEqual(result["claim"]["status"], "completed")
        self.assertTrue(result["claim"]["terminal"])
        self.assertFalse(result["claim"]["projection_applied"])
        self.assertFalse(result["claim"]["certification_applied"])
        self.assertTrue(
            all(
                criterion["phase"] == ActionPhase.ASSISTED.value
                for criterion in result["claim"]["evaluation"]["criteria"]
            )
        )
        self.assertEqual(self.failing_provider.calls, 1)
        report = self.ledger.report(self.attempt_id)
        self.assertEqual(report["evaluation_count"], 1)
        self.assertFalse(report["family_shadow_history"]["mastery_claim"])
        self.assertEqual(self._projection_snapshot(), self.before_projection)
        self.assertNotIn(self.private_scoring_key, raw)
        self.assertNotIn(str(self.database.path), raw)

    def test_exact_replay_does_not_add_lookup_callback_or_event(self) -> None:
        arguments = {
            "outcome": "unknown",
            "key": "reconcile-cli-replay",
        }
        first_code, first, _, first_error = self.reconcile_json(**arguments)
        self.assertEqual(first_code, 0, first_error)
        second_code, second, raw, second_error = self.reconcile_json(**arguments)
        self.assertEqual(second_code, 0, second_error)
        self.assertFalse(first["claim"]["idempotent_replay"])
        self.assertTrue(second["claim"]["idempotent_replay"])
        self.assertEqual(first["claim"]["reconciliation_count"], 1)
        self.assertEqual(second["claim"]["reconciliation_count"], 1)
        self.assert_boundaries(second)
        self.assertEqual(self.failing_provider.calls, 1)
        self.assertNotIn(self.private_scoring_key, raw)
        with self.database.read() as connection:
            observations = connection.execute(
                """SELECT COUNT(*) AS n
                   FROM performance_scoring_reconciliations
                   WHERE claim_id=?""",
                (self.claim_id,),
            ).fetchone()["n"]
            events = connection.execute(
                """SELECT COUNT(*) AS n FROM events
                   WHERE event_type='PerformanceScoringReconciled'
                     AND json_extract(payload_json, '$.claim_id')=?""",
                (self.claim_id,),
            ).fetchone()["n"]
        self.assertEqual(observations, 1)
        self.assertEqual(events, 1)
        self.assertEqual(self._projection_snapshot(), self.before_projection)

    def test_completed_replay_rejects_different_fixture_without_lookup(
        self,
    ) -> None:
        private_key = "reconcile-cli-completed-exact-fixture"
        first_code, first, _, first_error = self.reconcile_json(
            "completed",
            key=private_key,
            score="0.2",
            reliability="0.3",
        )
        self.assertEqual(first_code, 0, first_error)
        self.assertEqual(first["claim"]["status"], "completed")

        with patch(
            "tsq.cli.SyntheticReconciliationAdapter.lookup",
            side_effect=AssertionError("exact replay must not invoke lookup"),
        ):
            second_code, _, raw, second_error = self.reconcile_json(
                "completed",
                key=private_key,
                score="0.9",
                reliability="1.0",
            )
        self.assertEqual(second_code, 2)
        self.assertEqual(raw, "")
        self.assertIn(
            "different completed reconciliation fixture",
            second_error,
        )
        self.assertNotIn(private_key, second_error)
        self.assertNotIn(str(self.database.path), second_error)
        self.assertEqual(self.failing_provider.calls, 1)
        with self.database.read() as connection:
            self.assertEqual(
                connection.execute(
                    """SELECT COUNT(*) AS n
                       FROM performance_scoring_reconciliations
                       WHERE claim_id=?""",
                    (self.claim_id,),
                ).fetchone()["n"],
                1,
            )
            self.assertEqual(
                connection.execute(
                    """SELECT COUNT(*) AS n FROM task_evaluations
                       WHERE attempt_id=?""",
                    (self.attempt_id,),
                ).fetchone()["n"],
                1,
            )

    def test_reused_key_cannot_masquerade_as_a_different_outcome(self) -> None:
        private_key = "reconcile-cli-conflicting-private-key"
        first_code, first, _, first_error = self.reconcile_json(
            "unknown",
            key=private_key,
        )
        self.assertEqual(first_code, 0, first_error)
        self.assertEqual(first["claim"]["reconciliation_outcome"], "unknown")

        second_code, _, raw, second_error = self.reconcile_json(
            "definitely_absent",
            key=private_key,
        )
        self.assertEqual(second_code, 2)
        self.assertEqual(raw, "")
        self.assertIn(
            "different reconciliation outcome",
            second_error,
        )
        self.assertNotIn(private_key, second_error)
        self.assertEqual(self.failing_provider.calls, 1)
        claim = self.ledger.inspect_scoring_claim(self.claim_id)
        self.assertEqual(claim["status"], "unknown")
        self.assertEqual(claim["reconciliation_count"], 1)
        self.assertEqual(self._projection_snapshot(), self.before_projection)

    def test_plain_replay_distinguishes_observation_from_current_claim(
        self,
    ) -> None:
        unknown_key = "reconcile-cli-prior-unknown"
        unknown_code, _, _, unknown_error = self.reconcile_json(
            "unknown",
            key=unknown_key,
        )
        self.assertEqual(unknown_code, 0, unknown_error)
        completed_code, _, _, completed_error = self.reconcile_json(
            "completed",
            key="reconcile-cli-later-completed",
            score="0.7",
            reliability="0.8",
        )
        self.assertEqual(completed_code, 0, completed_error)

        with patch(
            "tsq.cli.SyntheticReconciliationAdapter.lookup",
            side_effect=AssertionError("exact replay must not invoke lookup"),
        ):
            code, plain, error = self.run_cli(
                "task",
                "reconcile-claim",
                self.claim_id,
                "--provider",
                "deterministic-test",
                "--test-outcome",
                "unknown",
                "--idempotency-key",
                unknown_key,
            )
        self.assertEqual(code, 0, error)
        self.assertIn("observation=unknown (idempotent replay)", plain)
        self.assertIn("claim=completed", plain)
        self.assertNotIn(unknown_key, plain)

    def test_invalid_outcome_options_fail_before_lookup_or_write(self) -> None:
        cases = (
            (
                "completed without score",
                ("completed", None, None),
                "--score is required",
            ),
            (
                "unknown score",
                ("unknown", "0.5", None),
                "valid only",
            ),
            (
                "absent reliability",
                ("definitely_absent", None, "0.5"),
                "valid only",
            ),
            (
                "score above one",
                ("completed", "1.1", None),
                "between 0 and 1",
            ),
            (
                "score nan",
                ("completed", "nan", None),
                "between 0 and 1",
            ),
            (
                "reliability below zero",
                ("completed", "0.5", "-0.1"),
                "between 0 and 1",
            ),
            (
                "reliability infinite",
                ("completed", "0.5", "inf"),
                "between 0 and 1",
            ),
        )
        for index, (label, values, message) in enumerate(cases):
            with self.subTest(label=label):
                code, _, output, error = self.reconcile_json(
                    values[0],
                    key=f"invalid-reconciliation-{index}",
                    score=values[1],
                    reliability=values[2],
                )
                self.assertEqual(code, 2)
                self.assertEqual(output, "")
                self.assertIn(message, error)
                self.assertNotIn("Traceback", error)
        claim = self.ledger.inspect_scoring_claim(self.claim_id)
        self.assertEqual(claim["status"], "unreconciled")
        self.assertEqual(claim["reconciliation_count"], 0)
        self.assertEqual(self.failing_provider.calls, 1)
        self.assertEqual(self._projection_snapshot(), self.before_projection)

    def test_real_provider_claim_is_ineligible_for_every_test_outcome(
        self,
    ) -> None:
        class FailingExternalProvider:
            provider_id = "provider.external-cli-fixture"
            provider_version = "v1"
            declared_kind = ScorerKind.MODEL
            synthetic = False

            def __init__(self) -> None:
                self.calls = 0

            def score(self, request):
                self.calls += 1
                raise RuntimeError("private external callback failure")

        provider = FailingExternalProvider()
        registry = ScoringProviderRegistry()
        registry.register(
            provider,
            ProviderAuthorityBinding(
                provider_id=provider.provider_id,
                provider_version=provider.provider_version,
                declared_kind=provider.declared_kind,
                authority_id="authority.external-cli-fixture",
                authority_manifest_digest="2" * 64,
                verified=False,
            ),
        )
        private_claim_key = "private-external-claim-key-never-render"
        with self.assertRaisesRegex(ValidationError, "failed safely"):
            self.ledger.score_attempt(
                self.attempt_id,
                registry,
                provider.provider_id,
                provider.provider_version,
                idempotency_key=private_claim_key,
            )
        claims = [
            claim
            for claim in self.ledger.list_scoring_claims(
                attempt_id=self.attempt_id,
                status="unreconciled",
            )
            if claim["provider_id"] == provider.provider_id
        ]
        self.assertEqual(len(claims), 1)
        external_claim_id = claims[0]["id"]
        before_rejections = self._projection_snapshot()

        cases = (
            ("unknown", None, None),
            ("definitely_absent", None, None),
            ("completed", "0.75", "0.8"),
        )
        with patch(
            "tsq.cli.SyntheticReconciliationAdapter.lookup",
            side_effect=AssertionError("synthetic lookup must not run"),
        ):
            for index, (outcome, score, reliability) in enumerate(cases):
                with self.subTest(outcome=outcome):
                    private_reconciliation_key = (
                        f"private-real-provider-reconciliation-{index}"
                    )
                    code, _, raw, error = self.reconcile_json(
                        outcome,
                        key=private_reconciliation_key,
                        score=score,
                        reliability=reliability,
                        claim_id=external_claim_id,
                    )
                    self.assertEqual(code, 2)
                    self.assertEqual(raw, "")
                    self.assertIn(
                        "only claims admitted by a synthetic scoring provider",
                        error,
                    )
                    self.assertNotIn(private_claim_key, error)
                    self.assertNotIn(private_reconciliation_key, error)
                    self.assertNotIn(str(self.database.path), error)
                    self.assertNotIn("provider_binding_digest", error)

        rejected = self.ledger.inspect_scoring_claim(external_claim_id)
        self.assertEqual(rejected["status"], "unreconciled")
        self.assertEqual(rejected["reconciliation_count"], 0)
        self.assertEqual(provider.calls, 1)
        with self.database.read() as connection:
            observations = connection.execute(
                """SELECT COUNT(*) AS n
                   FROM performance_scoring_reconciliations
                   WHERE claim_id=?""",
                (external_claim_id,),
            ).fetchone()["n"]
            events = connection.execute(
                """SELECT COUNT(*) AS n FROM events
                   WHERE event_type='PerformanceScoringReconciled'
                     AND json_extract(payload_json, '$.claim_id')=?""",
                (external_claim_id,),
            ).fetchone()["n"]
        self.assertEqual(observations, 0)
        self.assertEqual(events, 0)
        self.assertEqual(
            self.ledger.report(self.attempt_id)["evaluation_count"],
            0,
        )
        self.assertEqual(self._projection_snapshot(), before_rejections)


if __name__ == "__main__":
    unittest.main()
