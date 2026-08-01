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
from tsq.policy_shadow_reporting import (
    POLICY_SHADOW_REPORT_VERSION,
    PROSPECTIVE_ONE_STEP_OPE_VERSION,
    _Observation,
    _deterministic_challenger_ope,
    build_policy_shadow_report,
)
from tsq.store import Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
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
        self.assertEqual(
            report["report_version"], POLICY_SHADOW_REPORT_VERSION
        )
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
        self.assertEqual(
            report["prospective_shadow"]["one_step_ope"]["status"],
            "unavailable",
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
                """SELECT shadow.decision_id, shadow.agreement,
                          decision.propensity, attempt.is_correct
                   FROM policy_shadow_evaluations shadow
                   JOIN decisions decision
                     ON decision.id=shadow.decision_id
                   JOIN attempts attempt
                     ON attempt.decision_id=shadow.decision_id
                   WHERE shadow.decision_id IN (?, ?)
                   ORDER BY shadow.decision_id""",
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
        ope = prospective["one_step_ope"]

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
        self.assertTrue(outcomes["selection_conditioned"])
        self.assertFalse(outcomes["target_policy_estimate"])
        self.assertEqual(outcomes["raw_correct_count"], len(agreements))
        expected_credible = int(decision_ids[0] in agreements)
        self.assertEqual(
            outcomes["credible_retrieval_count"],
            expected_credible,
        )
        self.assertEqual(len(prospective["challengers"]), 1)
        self.assertFalse(prospective["response_censoring_adjusted"])
        self.assertEqual(
            ope["contract_version"], PROSPECTIVE_ONE_STEP_OPE_VERSION
        )
        self.assertEqual(ope["observation_count"], 2)
        self.assertEqual(
            ope["target_action_supported_count"], len(agreements)
        )
        self.assertFalse(ope["divergent_live_rewards_used"])
        self.assertFalse(ope["counterfactual_outcomes_imputed"])
        target_weights = [
            (1.0 / row["propensity"]) if row["agreement"] else 0.0
            for row in rows
        ]
        weighted_correct = sum(
            weight * row["is_correct"]
            for weight, row in zip(target_weights, rows, strict=True)
        )
        if sum(target_weights) == 0.0:
            self.assertEqual(ope["status"], "unavailable")
            self.assertIsNone(ope["raw_correctness"]["ips"])
            self.assertIsNone(ope["raw_correctness"]["snips"])
        else:
            self.assertAlmostEqual(
                ope["raw_correctness"]["ips"],
                weighted_correct / len(rows),
            )
            self.assertAlmostEqual(
                ope["raw_correctness"]["snips"],
                weighted_correct / sum(target_weights),
            )

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

    def test_shadow_frontier_projection_mismatch_fails_closed(self) -> None:
        session_id, presentation = self._start("shadow-mismatch")
        with self.database.transaction() as connection:
            connection.execute(
                "DROP TRIGGER policy_shadow_evaluations_no_update"
            )
            connection.execute(
                """UPDATE policy_shadow_evaluations
                   SET challenger_question_id=(
                       SELECT id FROM questions
                       WHERE id != (
                           SELECT challenger_question_id
                           FROM policy_shadow_evaluations
                           WHERE decision_id=?
                       )
                       ORDER BY id
                       LIMIT 1
                   )
                   WHERE decision_id=?""",
                (presentation.decision_id, presentation.decision_id),
            )

        with self.assertRaisesRegex(
            ValidationError, "event/projection integrity failed"
        ):
            build_policy_shadow_report(
                self.database, session_id=session_id
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
        self.assertEqual(
            versions["prospective_ope_contract_version"],
            PROSPECTIVE_ONE_STEP_OPE_VERSION,
        )

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
                ]
            )
        self.assertEqual(exit_code, 0)
        rendered = output.getvalue()
        self.assertIn("deterministic-challenger OPE", rendered)
        self.assertIn("not a policy estimate", rendered)


class DeterministicChallengerEstimatorTests(unittest.TestCase):
    @staticmethod
    def _observation(
        decision_id: str,
        *,
        propensity: float,
        correct: int,
        credible: int = 0,
    ) -> _Observation:
        return _Observation(
            decision_id=decision_id,
            prediction=0.5,
            correct=correct,
            credible_retrieval=credible,
            propensity=propensity,
            target_probability=0.0,
        )

    def test_hand_calculated_ips_snips_and_ess(self) -> None:
        rows = [
            {"decision_id": "d1", "agreement": 1},
            {"decision_id": "d2", "agreement": 0},
            {"decision_id": "d3", "agreement": 1},
        ]
        observations = {
            "d1": self._observation(
                "d1", propensity=0.5, correct=1, credible=1
            ),
            "d2": self._observation(
                "d2", propensity=0.4, correct=1, credible=1
            ),
            "d3": self._observation(
                "d3", propensity=0.25, correct=0, credible=0
            ),
        }

        result = _deterministic_challenger_ope(rows, observations)

        # Deterministic target weights are [1/.5, 0, 1/.25] = [2, 0, 4].
        self.assertEqual(result["weights"]["sum"], 6.0)
        self.assertAlmostEqual(
            result["weights"]["effective_sample_size"],
            36.0 / 20.0,
        )
        self.assertAlmostEqual(
            result["weights"]["effective_sample_ratio"],
            (36.0 / 20.0) / 3.0,
        )
        self.assertEqual(result["weights"]["zero_count"], 1)
        self.assertAlmostEqual(
            result["raw_correctness"]["ips"], 2.0 / 3.0
        )
        self.assertAlmostEqual(
            result["raw_correctness"]["snips"], 2.0 / 6.0
        )
        self.assertAlmostEqual(
            result["credible_retrieval"]["ips"], 2.0 / 3.0
        )
        self.assertAlmostEqual(
            result["credible_retrieval"]["snips"], 2.0 / 6.0
        )

    def test_divergent_reward_cannot_change_target_estimate(self) -> None:
        rows = [
            {"decision_id": "agree", "agreement": 1},
            {"decision_id": "diverge", "agreement": 0},
        ]
        common = {
            "agree": self._observation(
                "agree", propensity=0.5, correct=1
            ),
        }
        wrong_divergence = {
            **common,
            "diverge": self._observation(
                "diverge", propensity=0.5, correct=0
            ),
        }
        correct_divergence = {
            **common,
            "diverge": self._observation(
                "diverge", propensity=0.5, correct=1
            ),
        }

        wrong = _deterministic_challenger_ope(rows, wrong_divergence)
        correct = _deterministic_challenger_ope(rows, correct_divergence)

        self.assertEqual(
            wrong["raw_correctness"]["ips"],
            correct["raw_correctness"]["ips"],
        )
        self.assertEqual(
            wrong["raw_correctness"]["snips"],
            correct["raw_correctness"]["snips"],
        )
        self.assertNotEqual(
            wrong["raw_correctness"]["behavior_mean"],
            correct["raw_correctness"]["behavior_mean"],
        )

    def test_all_divergence_and_no_answers_are_unavailable(self) -> None:
        all_divergent = _deterministic_challenger_ope(
            [{"decision_id": "d1", "agreement": 0}],
            {
                "d1": self._observation(
                    "d1", propensity=0.2, correct=1
                )
            },
        )
        empty = _deterministic_challenger_ope(
            [{"decision_id": "pending", "agreement": 1}],
            {},
        )

        for result in (all_divergent, empty):
            self.assertEqual(result["status"], "unavailable")
            self.assertTrue(result["low_information"])
            self.assertIsNone(result["raw_correctness"]["ips"])
            self.assertIsNone(result["raw_correctness"]["snips"])
        self.assertEqual(
            all_divergent["low_information_reasons"],
            ["no answered target-action agreements"],
        )
        self.assertEqual(
            empty["low_information_reasons"],
            ["no answered shadow-evaluated decisions"],
        )

    def test_challenger_groups_remain_isolated(self) -> None:
        observations = {
            "a": self._observation("a", propensity=0.5, correct=1),
            "b": self._observation("b", propensity=0.5, correct=0),
        }
        first = _deterministic_challenger_ope(
            [
                {"decision_id": "a", "agreement": 1},
                {"decision_id": "b", "agreement": 0},
            ],
            observations,
        )
        second = _deterministic_challenger_ope(
            [
                {"decision_id": "a", "agreement": 0},
                {"decision_id": "b", "agreement": 1},
            ],
            observations,
        )

        self.assertEqual(first["raw_correctness"]["snips"], 1.0)
        self.assertEqual(second["raw_correctness"]["snips"], 0.0)


if __name__ == "__main__":
    unittest.main()
