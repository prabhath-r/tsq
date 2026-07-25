# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tsq.cli import main
from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.errors import ValidationError
from tsq.policy_shadow_reporting import build_policy_shadow_report
from tsq.store import Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"
START = datetime(2104, 6, 1, 9, 0, tzinfo=timezone.utc)


class PolicyShadowReportingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Database(
            Path(self.tempdir.name) / "policy-shadow-report.db"
        )
        self.database.initialize()
        self.database.import_corpus(
            *read_and_parse(CORPUS, include_catalog=True)
        )
        self.engine = AdaptiveEngine(self.database)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _start(
        self,
        learner_id: str = "report-learner",
        *,
        seed: int = 17,
    ) -> tuple[str, object]:
        self.engine.create_learner(learner_id)
        session = self.engine.start_session(
            learner_id,
            topic_id="t_llm_agents",
            mode="learn",
            seed=seed,
            now=START,
        )
        presentation = self.engine.next_question(session["id"], now=START)
        return session["id"], presentation

    def _two_answered_decisions(self) -> tuple[str, list[str]]:
        session_id, first = self._start()
        self.engine.submit_answer(
            first.decision_id,
            first.question.correct_option.id,
            confidence=0.90,
            response_ms=1_000,
            now=START + timedelta(seconds=2),
        )
        second = self.engine.next_question(
            session_id, now=START + timedelta(seconds=3)
        )
        self.engine.submit_answer(
            second.decision_id,
            second.question.correct_option.id,
            confidence=0.90,
            response_ms=100,
            now=START + timedelta(seconds=4),
        )
        return session_id, [first.decision_id, second.decision_id]

    def test_empty_report_is_explicitly_unavailable(self) -> None:
        before = self.database.path.read_bytes()
        report = build_policy_shadow_report(self.database)

        self.assertEqual(self.database.path.read_bytes(), before)
        self.assertEqual(report["decision_counts"]["total"], 0)
        self.assertEqual(report["behavior_calibration"]["count"], 0)
        self.assertIsNone(report["behavior_calibration"]["brier_score"])
        uniform = report["uniform_safe_frontier"]
        self.assertEqual(uniform["status"], "unavailable")
        self.assertTrue(uniform["low_information"])
        self.assertEqual(uniform["weights"]["count"], 0)
        self.assertEqual(
            report["prospective_shadow"]["evaluation_count"], 0
        )
        self.assertFalse(
            report["inference_boundary"]["counterfactual_trajectory"]
        )
        self.assertFalse(
            report["inference_boundary"]["causal_learning_effect"]
        )
        self.assertFalse(
            report["inference_boundary"]["retention_inference"]
        )
        self.assertTrue(
            report["inference_boundary"]["answered_complete_cases_only"]
        )
        self.assertFalse(
            report["inference_boundary"]["response_censoring_adjusted"]
        )
        self.assertFalse(
            report["inference_boundary"]["sequential_dependence_adjusted"]
        )

    def test_calibration_and_uniform_frontier_estimators_match_logged_data(
        self,
    ) -> None:
        session_id, _ = self._two_answered_decisions()

        report = build_policy_shadow_report(
            self.database, session_id=session_id
        )
        with self.database.read() as connection:
            rows = connection.execute(
                """SELECT decision.selected_score_json,
                          decision.propensity,
                          decision.candidate_count
                   FROM decisions decision
                   JOIN attempts attempt
                     ON attempt.decision_id=decision.id
                   WHERE decision.session_id=?
                   ORDER BY decision.created_at, decision.id""",
                (session_id,),
            ).fetchall()
        predictions = [
            json.loads(row["selected_score_json"])["predicted_correct"]
            for row in rows
        ]
        weights = [
            (1.0 / min(5, row["candidate_count"])) / row["propensity"]
            for row in rows
        ]
        expected_brier = sum((prediction - 1.0) ** 2 for prediction in predictions)
        expected_brier /= len(predictions)
        expected_log_loss = -sum(math.log(prediction) for prediction in predictions)
        expected_log_loss /= len(predictions)
        expected_ess = sum(weights) ** 2 / sum(
            weight * weight for weight in weights
        )

        calibration = report["behavior_calibration"]
        self.assertEqual(calibration["count"], 2)
        self.assertAlmostEqual(calibration["brier_score"], expected_brier)
        self.assertAlmostEqual(calibration["log_loss"], expected_log_loss)
        self.assertAlmostEqual(
            calibration["expected_calibration_error"],
            sum(abs(1.0 - prediction) for prediction in predictions)
            / len(predictions),
        )

        uniform = report["uniform_safe_frontier"]
        self.assertEqual(uniform["observation_count"], 2)
        self.assertTrue(uniform["low_information"])
        self.assertEqual(uniform["status"], "low_information")
        self.assertAlmostEqual(uniform["weights"]["sum"], sum(weights))
        self.assertAlmostEqual(
            uniform["weights"]["effective_sample_size"], expected_ess
        )
        self.assertAlmostEqual(
            uniform["weights"]["effective_sample_ratio"],
            expected_ess / len(weights),
        )
        self.assertEqual(uniform["weights"]["maximum"], max(weights))
        self.assertEqual(
            uniform["weights"]["p95"], sorted(weights)[-1]
        )
        self.assertEqual(uniform["weights"]["zero_count"], 0)
        self.assertEqual(uniform["weights"]["support_violations"], 0)
        self.assertIn(
            "does not assess independent sample size",
            uniform["information_scope"],
        )
        self.assertFalse(uniform["weights"]["dependence_adjusted"])
        self.assertEqual(
            uniform["raw_correctness"]["behavior_mean"], 1.0
        )
        self.assertAlmostEqual(
            uniform["raw_correctness"]["ips"],
            sum(weights) / len(weights),
        )
        self.assertEqual(
            uniform["raw_correctness"]["snips"], 1.0
        )
        # The second correct answer is too fast to be credible immediate
        # retrieval evidence under the immutable v8 response contract.
        self.assertEqual(
            uniform["credible_retrieval"]["behavior_mean"], 0.5
        )
        self.assertAlmostEqual(
            uniform["credible_retrieval"]["ips"],
            weights[0] / len(weights),
        )
        self.assertAlmostEqual(
            uniform["credible_retrieval"]["snips"],
            weights[0] / sum(weights),
        )

    def test_prospective_outcomes_use_only_observed_same_actions(self) -> None:
        session_id, decision_ids = self._two_answered_decisions()

        report = build_policy_shadow_report(
            self.database, session_id=session_id
        )
        with self.database.read() as connection:
            rows = connection.execute(
                """SELECT decision_id, agreement
                   FROM policy_shadow_evaluations
                   WHERE decision_id IN (?, ?)
                   ORDER BY decision_id""",
                tuple(decision_ids),
            ).fetchall()
        agreements = {
            row["decision_id"]
            for row in rows
            if row["agreement"] == 1
        }
        divergences = {
            row["decision_id"]
            for row in rows
            if row["agreement"] == 0
        }
        prospective = report["prospective_shadow"]
        outcomes = prospective["same_action_outcomes"]

        self.assertEqual(prospective["evaluation_count"], 2)
        self.assertEqual(prospective["agreement_count"], len(agreements))
        self.assertEqual(prospective["divergence_count"], len(divergences))
        self.assertEqual(
            prospective["divergent_observed_outcomes_withheld"],
            len(divergences),
        )
        self.assertEqual(
            outcomes["observed_same_action_count"], len(agreements)
        )
        self.assertEqual(outcomes["raw_correct_count"], len(agreements))
        expected_credible = int(decision_ids[0] in agreements)
        self.assertEqual(
            outcomes["credible_retrieval_count"],
            expected_credible,
        )
        self.assertEqual(len(prospective["challengers"]), 1)
        self.assertFalse(prospective["response_censoring_adjusted"])

    def test_prospective_agreement_reports_the_observed_outcome(self) -> None:
        session_id, presentation = self._start(
            "agreement-learner",
            seed=2,
        )
        self.engine.submit_answer(
            presentation.decision_id,
            presentation.question.correct_option.id,
            confidence=0.90,
            response_ms=1_000,
            now=START + timedelta(seconds=2),
        )

        report = build_policy_shadow_report(
            self.database, session_id=session_id
        )
        prospective = report["prospective_shadow"]
        outcomes = prospective["same_action_outcomes"]
        self.assertEqual(prospective["evaluation_count"], 1)
        self.assertEqual(prospective["agreement_count"], 1)
        self.assertEqual(prospective["divergence_count"], 0)
        self.assertEqual(outcomes["observed_same_action_count"], 1)
        self.assertEqual(outcomes["raw_correct_count"], 1)
        self.assertEqual(outcomes["credible_retrieval_count"], 1)

    def test_incorrect_outcome_uses_complement_log_probability(self) -> None:
        session_id, presentation = self._start("incorrect-calibration")
        wrong_option = next(
            option
            for option in presentation.question.options
            if not option.correct
        )
        self.engine.submit_answer(
            presentation.decision_id,
            wrong_option.id,
            confidence=0.80,
            response_ms=1_000,
            now=START + timedelta(seconds=2),
        )

        report = build_policy_shadow_report(
            self.database, session_id=session_id
        )
        with self.database.read() as connection:
            selected_score = json.loads(
                connection.execute(
                    """SELECT selected_score_json FROM decisions
                       WHERE id=?""",
                    (presentation.decision_id,),
                ).fetchone()["selected_score_json"]
            )
        prediction = selected_score["predicted_correct"]
        calibration = report["behavior_calibration"]
        self.assertEqual(calibration["count"], 1)
        self.assertAlmostEqual(
            calibration["brier_score"],
            prediction * prediction,
        )
        self.assertAlmostEqual(
            calibration["log_loss"],
            -math.log(1.0 - prediction),
        )
        self.assertEqual(
            report["uniform_safe_frontier"]["raw_correctness"][
                "behavior_mean"
            ],
            0.0,
        )

    def test_pending_and_invalidated_decisions_remain_complete_cases_only(
        self,
    ) -> None:
        session_id, _presentation = self._start("censoring-boundary")

        pending = build_policy_shadow_report(
            self.database, session_id=session_id
        )
        self.assertEqual(
            pending["decision_counts"],
            {
                "total": 1,
                "answered": 0,
                "pending": 1,
                "invalidated": 0,
            },
        )
        self.assertEqual(
            pending["uniform_safe_frontier"]["status"],
            "unavailable",
        )

        self.engine.end_session(
            session_id,
            status="abandoned",
            reason="test_invalidation",
            idempotency_key="report-invalidate-pending",
            now=START + timedelta(seconds=2),
        )
        invalidated = build_policy_shadow_report(
            self.database, session_id=session_id
        )
        self.assertEqual(
            invalidated["decision_counts"],
            {
                "total": 1,
                "answered": 0,
                "pending": 0,
                "invalidated": 1,
            },
        )
        self.assertTrue(
            invalidated["inference_boundary"][
                "answered_complete_cases_only"
            ]
        )

    def test_unknown_selected_score_field_fails_closed(self) -> None:
        _, presentation = self._start("malformed-score")
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT selected_score_json FROM decisions WHERE id=?",
                (presentation.decision_id,),
            ).fetchone()
            score = json.loads(row["selected_score_json"])
            score["future_unversioned_field"] = 1
            connection.execute(
                "UPDATE decisions SET selected_score_json=? WHERE id=?",
                (json.dumps(score), presentation.decision_id),
            )

        with self.assertRaisesRegex(ValidationError, "unknown"):
            build_policy_shadow_report(
                self.database, session_id=presentation.session_id
            )

    def test_unknown_logging_policy_fails_closed(self) -> None:
        _, presentation = self._start("unknown-policy")
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE decisions SET policy_version=? WHERE id=?",
                ("future-unreviewed-policy", presentation.decision_id),
            )

        with self.assertRaisesRegex(
            ValidationError, "unsupported logging policy"
        ):
            build_policy_shadow_report(
                self.database, session_id=presentation.session_id
            )

    def test_cli_report_and_versions_expose_shadow_boundaries(self) -> None:
        session_id, _ = self._two_answered_decisions()
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                [
                    "--db",
                    str(self.database.path),
                    "policy",
                    "report",
                    "--session",
                    session_id,
                    "--json",
                ]
            )
        self.assertEqual(exit_code, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["decision_counts"]["answered"], 2)
        self.assertTrue(report["inference_boundary"]["shadow_only"])
        self.assertFalse(
            report["inference_boundary"]["causal_learning_effect"]
        )

        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(["policy", "versions", "--json"])
        self.assertEqual(exit_code, 0)
        versions = json.loads(output.getvalue())
        self.assertFalse(versions["inference_boundary"]["selection_applied"])
        self.assertFalse(versions["inference_boundary"]["mastery_applied"])
        self.assertFalse(versions["inference_boundary"]["causal_claim"])


if __name__ == "__main__":
    unittest.main()
