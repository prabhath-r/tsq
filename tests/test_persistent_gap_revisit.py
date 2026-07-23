# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from tsq.errors import ValidationError
from tsq.models import SessionPhase
from tsq.policy import (
    POLICY_VERSION,
    PERSISTENT_GAP_EPISODE_BUDGET,
    AdaptivePolicy,
)


NOW = datetime(2105, 4, 8, 9, 0, tzinfo=timezone.utc)


def objective_state(
    objective_id: str,
    *,
    mastery: float,
    due_at: datetime | None,
    error: float = 5e-5,
) -> SimpleNamespace:
    return SimpleNamespace(
        objective_id=objective_id,
        mastery_probability=mastery,
        mastery_probability_error_bound=error,
        next_review_at=due_at,
    )


class PersistentGapRevisitTests(unittest.TestCase):
    def test_due_independent_below_prior_gap_joins_breadth_frontier(self) -> None:
        cold = objective_state(
            "lo_gap",
            mastery=0.10,
            due_at=None,
        )
        projected = objective_state(
            "lo_gap",
            mastery=0.04,
            due_at=NOW - timedelta(hours=1),
        )
        self.assertTrue(
            AdaptivePolicy._is_due_persistent_gap(
                projected,
                cold,
                observed_response_families=2,
                now=NOW,
            )
        )

        frontier = SimpleNamespace(id="q_frontier", objective_id="lo_new")
        gap = SimpleNamespace(id="q_gap", objective_id="lo_gap")
        candidates, minimum, bypassed = (
            AdaptivePolicy._fair_coverage_candidates(
                (frontier, gap),
                target_exposures={"q_frontier": 0, "q_gap": 5},
                persistent_gap_objective_ids={"lo_gap"},
            )
        )

        self.assertEqual(minimum, 0)
        self.assertEqual(
            {question.id for question in candidates},
            {"q_frontier", "q_gap"},
        )
        self.assertEqual(bypassed, {"lo_gap"})

    def test_strong_or_not_due_objectives_do_not_bypass_breadth(self) -> None:
        cases = {
            "lo_strong": objective_state(
                "lo_strong",
                mastery=0.11,
                due_at=NOW - timedelta(days=1),
            ),
            "lo_not_due": objective_state(
                "lo_not_due",
                mastery=0.02,
                due_at=NOW + timedelta(minutes=1),
            ),
            "lo_numerical_tie": objective_state(
                "lo_numerical_tie",
                mastery=0.09997,
                due_at=NOW - timedelta(days=1),
                error=2e-5,
            ),
        }
        qualified = {
            objective_id
            for objective_id, projected in cases.items()
            if AdaptivePolicy._is_due_persistent_gap(
                projected,
                objective_state(
                    objective_id,
                    mastery=0.10,
                    due_at=None,
                    error=2e-5,
                ),
                observed_response_families=2,
                now=NOW,
            )
        }
        self.assertEqual(qualified, set())

        questions = (
            SimpleNamespace(id="q_frontier", objective_id="lo_new"),
            SimpleNamespace(id="q_strong", objective_id="lo_strong"),
            SimpleNamespace(id="q_not_due", objective_id="lo_not_due"),
        )
        candidates, minimum, bypassed = (
            AdaptivePolicy._fair_coverage_candidates(
                questions,
                target_exposures={
                    "q_frontier": 0,
                    "q_strong": 4,
                    "q_not_due": 3,
                },
                persistent_gap_objective_ids=qualified,
            )
        )
        self.assertEqual(minimum, 0)
        self.assertEqual([question.id for question in candidates], ["q_frontier"])
        self.assertEqual(bypassed, set())

    def test_one_repeated_family_is_not_independent_evidence(self) -> None:
        cold = objective_state("lo_gap", mastery=0.10, due_at=None)
        projected = objective_state(
            "lo_gap",
            mastery=0.01,
            due_at=NOW - timedelta(days=1),
        )
        self.assertFalse(
            AdaptivePolicy._is_due_persistent_gap(
                projected,
                cold,
                observed_response_families=1,
                now=NOW,
            )
        )

    def test_naive_due_timestamp_fails_closed(self) -> None:
        cold = objective_state("lo_gap", mastery=0.10, due_at=None)
        projected = objective_state(
            "lo_gap",
            mastery=0.01,
            due_at=datetime(2105, 4, 8, 8, 0),
        )
        with self.assertRaisesRegex(ValidationError, "timezone-aware"):
            AdaptivePolicy._is_due_persistent_gap(
                projected,
                cold,
                observed_response_families=2,
                now=NOW,
            )

    def test_breadth_bypass_is_explicit_in_persisted_rationale(self) -> None:
        question = SimpleNamespace(
            objective_id="lo_gap",
            primary_concept_id="c_gap",
            misconception_ids=frozenset(),
        )
        score = SimpleNamespace(
            predicted_correct=0.35,
            information_gain=0.42,
            concept_need=0.80,
            review_value=0.75,
            boundary_fit=1.0,
            continuity=0.40,
        )

        rationale = AdaptivePolicy._rationale(
            question,
            score,
            SessionPhase.LEARN,
            None,
            None,
            fair_coverage_exposure=0,
            persistent_gap_revisit={
                "observed_response_families": 3,
                "mastery_probability": 0.04,
                "cold_start_mastery_probability": 0.10,
                "next_review_at": NOW - timedelta(hours=2),
                "episode_spend": 1,
                "episode_budget": PERSISTENT_GAP_EPISODE_BUDGET,
            },
        )

        self.assertIn("persistent_gap_revisit=lo_gap", rationale)
        self.assertIn("persistent_gap_observed_families=3", rationale)
        self.assertIn("persistent_gap_mastery=0.040000", rationale)
        self.assertIn("persistent_gap_cold_prior=0.100000", rationale)
        self.assertIn("persistent_gap_episode_spend=1", rationale)
        self.assertIn("persistent_gap_episode_budget=2", rationale)
        self.assertIn("fair_coverage_target_exposures=0", rationale)
        marker = AdaptivePolicy._persistent_gap_marker(
            rationale=rationale,
            policy_version=POLICY_VERSION,
            decision_objective_id="lo_gap",
        )
        self.assertIsNotNone(marker)
        self.assertEqual(marker["spend"], 1)
        self.assertEqual(marker["budget"], 2)
        self.assertEqual(
            rationale,
            AdaptivePolicy._rationale(
                question,
                score,
                SessionPhase.LEARN,
                None,
                None,
                fair_coverage_exposure=0,
                persistent_gap_revisit={
                    "observed_response_families": 3,
                    "mastery_probability": 0.04,
                    "cold_start_mastery_probability": 0.10,
                    "next_review_at": NOW - timedelta(hours=2),
                    "episode_spend": 1,
                    "episode_budget": PERSISTENT_GAP_EPISODE_BUDGET,
                },
            ),
        )

    def test_episode_budget_requires_interleaving_and_closes_early(self) -> None:
        next_spend = AdaptivePolicy._next_persistent_gap_episode_spend
        self.assertEqual(
            next_spend(
                prior_spends=0,
                gap_open=True,
                due=True,
                interleaved=False,
                distinct_capacity=True,
            ),
            1,
        )
        self.assertIsNone(
            next_spend(
                prior_spends=1,
                gap_open=True,
                due=False,
                interleaved=False,
                distinct_capacity=True,
            )
        )
        self.assertEqual(
            next_spend(
                prior_spends=1,
                gap_open=True,
                due=False,
                interleaved=True,
                distinct_capacity=True,
            ),
            2,
        )
        for state in (
            {
                "prior_spends": 1,
                "gap_open": False,
                "due": False,
                "interleaved": True,
                "distinct_capacity": True,
            },
            {
                "prior_spends": 1,
                "gap_open": True,
                "due": False,
                "interleaved": True,
                "distinct_capacity": False,
            },
            {
                "prior_spends": 2,
                "gap_open": True,
                "due": True,
                "interleaved": True,
                "distinct_capacity": True,
            },
        ):
            self.assertIsNone(next_spend(**state), state)

    def test_current_markers_fail_closed_when_malformed(self) -> None:
        valid = (
            "persistent_gap_revisit=lo_gap; "
            "persistent_gap_observed_families=2; "
            "persistent_gap_mastery=0.040000; "
            "persistent_gap_cold_prior=0.100000; "
            f"persistent_gap_due_at={NOW.isoformat()}; "
            "persistent_gap_episode_spend=1; "
            "persistent_gap_episode_budget=2"
        )
        cases = (
            valid.replace("; persistent_gap_episode_budget=2", ""),
            valid.replace(
                "persistent_gap_episode_budget=2",
                "persistent_gap_episode_budget=3",
            ),
            valid + "; persistent_gap_surprise=yes",
            valid.replace(
                "persistent_gap_revisit=lo_gap",
                "persistent_gap_revisit=lo_other",
            ),
        )
        for rationale in cases:
            with self.subTest(rationale=rationale):
                with self.assertRaises(ValidationError):
                    AdaptivePolicy._persistent_gap_marker(
                        rationale=rationale,
                        policy_version=POLICY_VERSION,
                        decision_objective_id="lo_gap",
                    )

    def test_complete_v12_marker_can_seed_one_upgrade_spend(self) -> None:
        rationale = (
            "persistent_gap_revisit=lo_gap; "
            "persistent_gap_observed_families=2; "
            "persistent_gap_mastery=0.040000; "
            "persistent_gap_cold_prior=0.100000; "
            f"persistent_gap_due_at={NOW.isoformat()}"
        )
        marker = AdaptivePolicy._persistent_gap_marker(
            rationale=rationale,
            policy_version="recursive-evidence-graph-v12",
            decision_objective_id="lo_gap",
        )
        self.assertEqual(marker["spend"], 1)
        self.assertEqual(marker["budget"], 2)


if __name__ == "__main__":
    unittest.main()
