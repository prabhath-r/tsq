# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import sqlite3
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from tsq.inference import (
    ResponseClass,
    classify_response_for_model,
    credible_response_sql,
    response_window,
)
from tsq.objective_posterior import (
    OBJECTIVE_POSTERIOR_ALGORITHM,
    OBJECTIVE_POSTERIOR_CODEC,
    OBJECTIVE_POSTERIOR_GRID_ID,
    OBJECTIVE_POSTERIOR_SCHEMA_VERSION,
    OBJECTIVE_POSTERIOR_V1_IDENTITY,
    SUPPORTED_OBJECTIVE_POSTERIOR_IDENTITIES,
)
from tsq.versions import (
    AUTHORITATIVE_RESPONSE_WINDOW_MODEL_VERSIONS,
    CONCEPT_MODEL_VERSION,
    LEGACY_MODEL_VERSION,
    OBJECTIVE_GAUSSIAN_MODEL_VERSION,
    OBJECTIVE_GRID_V6_MODEL_VERSION,
    OBJECTIVE_GRID_V7_MODEL_VERSION,
    OBJECTIVE_GRID_V8_MODEL_VERSION,
    OBJECTIVE_PROJECTION_FORMATS,
    RESPONSE_TELEMETRY_CONTRACTS,
    SUPPORTED_MODEL_VERSIONS,
)


HISTORICAL_OPTIONAL_MODELS = (
    LEGACY_MODEL_VERSION,
    CONCEPT_MODEL_VERSION,
    OBJECTIVE_GAUSSIAN_MODEL_VERSION,
)
KNOWN_MODELS = (
    *HISTORICAL_OPTIONAL_MODELS,
    OBJECTIVE_GRID_V6_MODEL_VERSION,
    OBJECTIVE_GRID_V7_MODEL_VERSION,
    OBJECTIVE_GRID_V8_MODEL_VERSION,
)


class ResponseVersionSemanticsTests(unittest.TestCase):
    def _classify(
        self,
        model_version: str,
        *,
        correct: bool = True,
        selected_option_id: str | None = "option-a",
        selected_misconception_id: str | None = None,
        confidence: float | None = 0.9,
        response_ms: int | None = 500,
        hint_count: int = 0,
    ) -> ResponseClass:
        return classify_response_for_model(
            model_version=model_version,
            correct=correct,
            selected_option_id=selected_option_id,
            selected_misconception_id=selected_misconception_id,
            confidence=confidence,
            response_ms=response_ms,
            hint_count=hint_count,
        )

    def _sql_credible(
        self,
        model_version: str,
        *,
        selected_option_id: str | None = "option-a",
        confidence: float | None = 0.9,
        response_ms: int | None = 500,
        hint_count: int = 0,
    ) -> bool:
        clause = credible_response_sql(
            model_expression="attempt.model_version",
        )
        connection = sqlite3.connect(":memory:")
        try:
            row = connection.execute(
                f"""WITH attempt(
                        model_version, selected_option_id, confidence,
                        response_ms, hint_count
                    ) AS (VALUES (?, ?, ?, ?, ?))
                    SELECT ({clause}) AS credible FROM attempt""",
                (
                    model_version,
                    selected_option_id,
                    confidence,
                    response_ms,
                    hint_count,
                ),
            ).fetchone()
        finally:
            connection.close()
        return bool(row[0])

    def test_classifier_and_routing_sql_share_the_version_truth_table(
        self,
    ) -> None:
        cases = (
            (None, None, 0),
            (0.9, None, 0),
            (None, 500, 0),
            (0.9, 500, 0),
            (0.49, 500, 0),
            (0.9, 249, 0),
            (0.9, 500, 1),
        )
        for model_version in (*KNOWN_MODELS, "future-model-v99"):
            for confidence, response_ms, hint_count in cases:
                with self.subTest(
                    model=model_version,
                    confidence=confidence,
                    response_ms=response_ms,
                    hint_count=hint_count,
                ):
                    response_class = self._classify(
                        model_version,
                        confidence=confidence,
                        response_ms=response_ms,
                        hint_count=hint_count,
                    )
                    self.assertEqual(
                        self._sql_credible(
                            model_version,
                            confidence=confidence,
                            response_ms=response_ms,
                            hint_count=hint_count,
                        ),
                        response_class is ResponseClass.CREDIBLE_SUCCESS,
                    )

        for model_version in KNOWN_MODELS:
            self.assertFalse(
                self._sql_credible(
                    model_version,
                    selected_option_id=None,
                )
            )

    def test_historical_optional_fields_still_fail_on_present_weak_values(
        self,
    ) -> None:
        for model_version in HISTORICAL_OPTIONAL_MODELS:
            self.assertIs(
                self._classify(
                    model_version,
                    confidence=None,
                    response_ms=None,
                ),
                ResponseClass.CREDIBLE_SUCCESS,
            )
            self.assertIs(
                self._classify(
                    model_version,
                    confidence=0.49,
                    response_ms=None,
                ),
                ResponseClass.NONCREDIBLE_SUCCESS,
            )
            self.assertIs(
                self._classify(
                    model_version,
                    confidence=None,
                    response_ms=249,
                ),
                ResponseClass.NONCREDIBLE_SUCCESS,
            )

    def test_v6_requires_time_and_v7_v8_require_both_observations(self) -> None:
        self.assertIs(
            self._classify(
                OBJECTIVE_GRID_V6_MODEL_VERSION,
                confidence=None,
                response_ms=500,
            ),
            ResponseClass.CREDIBLE_SUCCESS,
        )
        self.assertIs(
            self._classify(
                OBJECTIVE_GRID_V6_MODEL_VERSION,
                confidence=0.9,
                response_ms=None,
            ),
            ResponseClass.NONCREDIBLE_SUCCESS,
        )
        for model_version in (
            OBJECTIVE_GRID_V7_MODEL_VERSION,
            OBJECTIVE_GRID_V8_MODEL_VERSION,
        ):
            for confidence, response_ms in ((None, 500), (0.9, None)):
                self.assertIs(
                    self._classify(
                        model_version,
                        confidence=confidence,
                        response_ms=response_ms,
                    ),
                    ResponseClass.NONCREDIBLE_SUCCESS,
                )

    def test_v7_and_v8_require_high_confidence_for_a_named_error(self) -> None:
        for model_version in (
            *HISTORICAL_OPTIONAL_MODELS,
            OBJECTIVE_GRID_V6_MODEL_VERSION,
        ):
            self.assertIs(
                self._classify(
                    model_version,
                    correct=False,
                    selected_misconception_id="m_named",
                    confidence=0.5,
                ),
                ResponseClass.CREDIBLE_NAMED_ERROR,
            )
        self.assertIs(
            self._classify(
                OBJECTIVE_GRID_V7_MODEL_VERSION,
                correct=False,
                selected_misconception_id="m_named",
                confidence=0.79,
            ),
            ResponseClass.CREDIBLE_GENERIC_ERROR,
        )
        self.assertIs(
            self._classify(
                OBJECTIVE_GRID_V7_MODEL_VERSION,
                correct=False,
                selected_misconception_id="m_named",
                confidence=0.80,
            ),
            ResponseClass.CREDIBLE_NAMED_ERROR,
        )
        self.assertIs(
            self._classify(
                OBJECTIVE_GRID_V8_MODEL_VERSION,
                correct=False,
                selected_misconception_id="m_named",
                confidence=0.79,
            ),
            ResponseClass.CREDIBLE_GENERIC_ERROR,
        )
        self.assertIs(
            self._classify(
                OBJECTIVE_GRID_V8_MODEL_VERSION,
                correct=False,
                selected_misconception_id="m_named",
                confidence=0.80,
            ),
            ResponseClass.CREDIBLE_NAMED_ERROR,
        )

    def test_version_registry_and_posterior_v1_identity_are_explicit(self) -> None:
        self.assertEqual(set(RESPONSE_TELEMETRY_CONTRACTS), set(KNOWN_MODELS))
        self.assertEqual(set(SUPPORTED_MODEL_VERSIONS), set(KNOWN_MODELS))
        self.assertEqual(
            AUTHORITATIVE_RESPONSE_WINDOW_MODEL_VERSIONS,
            frozenset({OBJECTIVE_GRID_V8_MODEL_VERSION}),
        )
        self.assertEqual(
            (
                OBJECTIVE_PROJECTION_FORMATS[
                    OBJECTIVE_GRID_V6_MODEL_VERSION
                ].event_schema_version,
                OBJECTIVE_PROJECTION_FORMATS[
                    OBJECTIVE_GRID_V6_MODEL_VERSION
                ].hash_version,
            ),
            (4, 3),
        )
        self.assertEqual(
            SUPPORTED_OBJECTIVE_POSTERIOR_IDENTITIES,
            frozenset({OBJECTIVE_POSTERIOR_V1_IDENTITY}),
        )
        self.assertEqual(
            (
                OBJECTIVE_POSTERIOR_SCHEMA_VERSION,
                OBJECTIVE_POSTERIOR_ALGORITHM,
                OBJECTIVE_POSTERIOR_GRID_ID,
                OBJECTIVE_POSTERIOR_CODEC,
            ),
            (
                OBJECTIVE_POSTERIOR_V1_IDENTITY.schema_version,
                OBJECTIVE_POSTERIOR_V1_IDENTITY.algorithm,
                OBJECTIVE_POSTERIOR_V1_IDENTITY.grid_id,
                OBJECTIVE_POSTERIOR_V1_IDENTITY.codec,
            ),
        )

    def test_future_default_alias_cannot_reinterpret_historical_events(
        self,
    ) -> None:
        with patch("tsq.learner.MODEL_VERSION", "future-model-v99"):
            self.assertIs(
                self._classify(
                    OBJECTIVE_GRID_V6_MODEL_VERSION,
                    confidence=None,
                    response_ms=500,
                ),
                ResponseClass.CREDIBLE_SUCCESS,
            )
            self.assertEqual(
                OBJECTIVE_PROJECTION_FORMATS[
                    OBJECTIVE_GRID_V6_MODEL_VERSION
                ].event_schema_version,
                4,
            )

    def test_authoritative_window_uses_exact_integer_milliseconds(self) -> None:
        selected_at = datetime(2100, 1, 1, tzinfo=timezone.utc)
        for response_ms in (1, 249, 3_634, 60_001):
            with self.subTest(response_ms=response_ms):
                window = response_window(
                    selected_at=selected_at,
                    answered_at=selected_at
                    + timedelta(milliseconds=response_ms),
                    response_ms=response_ms,
                )
                self.assertEqual(window.elapsed_ms, response_ms)
                self.assertTrue(window.consistent)


if __name__ == "__main__":
    unittest.main()
