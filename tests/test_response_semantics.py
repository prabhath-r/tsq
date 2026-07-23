# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from tsq.inference import ResponseClass, classify_response, response_window


START = datetime(2120, 1, 1, 9, 0, tzinfo=timezone.utc)


class ResponseSemanticsTests(unittest.TestCase):
    def classify(
        self,
        *,
        correct: bool,
        selected: bool = True,
        named: bool = False,
        confidence: float | None = 0.9,
        response_ms: int | None = 900,
        hint_count: int = 0,
    ) -> ResponseClass:
        return classify_response(
            correct=correct,
            selected_option_id="option" if selected else None,
            selected_misconception_id="misconception" if named else None,
            confidence=confidence,
            response_ms=response_ms,
            hint_count=hint_count,
        )

    def test_only_observable_unguided_success_certifies(self) -> None:
        self.assertEqual(
            self.classify(correct=True), ResponseClass.CREDIBLE_SUCCESS
        )
        for kwargs in (
            {"confidence": None},
            {"confidence": 0.49},
            {"response_ms": None},
            {"response_ms": 249},
            {"hint_count": 1},
        ):
            with self.subTest(**kwargs):
                self.assertEqual(
                    self.classify(correct=True, **kwargs),
                    ResponseClass.NONCREDIBLE_SUCCESS,
                )

    def test_named_error_requires_stronger_self_report(self) -> None:
        self.assertEqual(
            self.classify(correct=False, named=True, confidence=0.8),
            ResponseClass.CREDIBLE_NAMED_ERROR,
        )
        self.assertEqual(
            self.classify(correct=False, named=True, confidence=0.79),
            ResponseClass.CREDIBLE_GENERIC_ERROR,
        )
        self.assertEqual(
            self.classify(correct=False, named=False, confidence=0.9),
            ResponseClass.CREDIBLE_GENERIC_ERROR,
        )

    def test_omission_uncertainty_and_assistance_do_not_localize_failure(self) -> None:
        for kwargs in (
            {"selected": False},
            {"confidence": None},
            {"confidence": 0.49},
            {"response_ms": None},
            {"response_ms": 249},
            {"hint_count": 1},
        ):
            with self.subTest(**kwargs):
                self.assertEqual(
                    self.classify(correct=False, named=True, **kwargs),
                    ResponseClass.UNCERTAIN_OR_ABSTAINED,
                )

    def test_claimed_active_time_cannot_exceed_authoritative_window(self) -> None:
        consistent = response_window(
            selected_at=START,
            answered_at=START + timedelta(milliseconds=900),
            response_ms=900,
        )
        self.assertTrue(consistent.consistent)
        self.assertEqual(consistent.elapsed_ms, 900)
        impossible = response_window(
            selected_at=START,
            answered_at=START,
            response_ms=4_000,
        )
        self.assertFalse(impossible.consistent)

    def test_response_window_rejects_backward_or_naive_time(self) -> None:
        with self.assertRaisesRegex(ValueError, "precede"):
            response_window(
                selected_at=START,
                answered_at=START - timedelta(microseconds=1),
                response_ms=0,
            )
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            response_window(
                selected_at=START.replace(tzinfo=None),
                answered_at=START,
                response_ms=0,
            )


if __name__ == "__main__":
    unittest.main()
