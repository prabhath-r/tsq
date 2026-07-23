# SPDX-License-Identifier: MPL-2.0

"""Stable response semantics used by learning, routing, and replay.

Raw correctness is not a sufficient pedagogical observation.  A keyed answer
can be a credible retrieval, an uncertain success, a named misconception, or
only generic evidence of difficulty.  Keeping that interpretation in one
small, deterministic boundary prevents the learner update and routing policy
from silently assigning different meanings to the same immutable response.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .versions import (
    RESPONSE_TELEMETRY_CONTRACTS,
    ResponseTelemetryContract,
)

MISCONCEPTION_ALGORITHM_METADATA_KEY = "misconception_algorithm"
LEGACY_MISCONCEPTION_ALGORITHM = "legacy-additive-v1"
MISCONCEPTION_ALGORITHM_VERSION = "family-latest-credible-signed-v1"
SUPPORTED_MISCONCEPTION_ALGORITHMS = frozenset(
    {LEGACY_MISCONCEPTION_ALGORITHM, MISCONCEPTION_ALGORITHM_VERSION}
)


RESPONSE_INTERPRETATION_VERSION = "behavioral-credibility-v1"
MIN_CREDIBLE_RESPONSE_MS = 250
MIN_CREDIBLE_CONFIDENCE = 0.50
MIN_NAMED_ERROR_CONFIDENCE = 0.80


class ResponseClass(str, Enum):
    """Mutually exclusive meaning of one answer observation."""

    CREDIBLE_SUCCESS = "credible_success"
    NONCREDIBLE_SUCCESS = "noncredible_success"
    CREDIBLE_NAMED_ERROR = "credible_named_error"
    CREDIBLE_GENERIC_ERROR = "credible_generic_error"
    UNCERTAIN_OR_ABSTAINED = "uncertain_or_abstained"

    @property
    def certifies_retrieval(self) -> bool:
        return self is ResponseClass.CREDIBLE_SUCCESS

    @property
    def supports_failure_localization(self) -> bool:
        return self in {
            ResponseClass.CREDIBLE_NAMED_ERROR,
            ResponseClass.CREDIBLE_GENERIC_ERROR,
        }

    @property
    def supports_named_misconception(self) -> bool:
        return self is ResponseClass.CREDIBLE_NAMED_ERROR


def classify_response(
    *,
    correct: bool,
    selected_option_id: str | None,
    selected_misconception_id: str | None,
    confidence: float | None,
    response_ms: int | None,
    hint_count: int,
) -> ResponseClass:
    """Classify validated immutable answer facts, failing closed on omissions.

    A missing confidence or response time can still update uncertain mastery,
    but it cannot certify knowledge or localize a failure.  A moderately
    confident wrong selection is useful generic evidence; the stronger named
    misconception claim requires the higher confidence threshold.
    """

    observable_and_unguided = bool(
        selected_option_id is not None
        and hint_count == 0
        and confidence is not None
        and response_ms is not None
        and response_ms >= MIN_CREDIBLE_RESPONSE_MS
    )
    if correct:
        if (
            observable_and_unguided
            and confidence is not None
            and confidence >= MIN_CREDIBLE_CONFIDENCE
        ):
            return ResponseClass.CREDIBLE_SUCCESS
        return ResponseClass.NONCREDIBLE_SUCCESS
    if (
        not observable_and_unguided
        or confidence is None
        or confidence < MIN_CREDIBLE_CONFIDENCE
    ):
        return ResponseClass.UNCERTAIN_OR_ABSTAINED
    if (
        selected_misconception_id is not None
        and confidence >= MIN_NAMED_ERROR_CONFIDENCE
    ):
        return ResponseClass.CREDIBLE_NAMED_ERROR
    return ResponseClass.CREDIBLE_GENERIC_ERROR


def response_telemetry_contract(
    model_version: object,
) -> ResponseTelemetryContract | None:
    """Return one immutable model's contract, or ``None`` for unknown models."""

    return (
        RESPONSE_TELEMETRY_CONTRACTS.get(model_version)
        if isinstance(model_version, str)
        else None
    )


def classify_response_for_model(
    *,
    model_version: object,
    correct: bool,
    selected_option_id: str | None,
    selected_misconception_id: str | None,
    confidence: float | None,
    response_ms: int | None,
    hint_count: int,
) -> ResponseClass:
    """Interpret an answer using the contract named by its immutable event.

    Unknown model versions are deliberately noncredible.  Replay and integrity
    reject them separately; routing must still fail closed if it encounters one
    before that audit boundary.
    """

    contract = response_telemetry_contract(model_version)
    if contract is None:
        return (
            ResponseClass.NONCREDIBLE_SUCCESS
            if correct
            else ResponseClass.UNCERTAIN_OR_ABSTAINED
        )
    observable_and_unguided = bool(
        selected_option_id is not None
        and hint_count == 0
        and (
            confidence is not None
            and confidence >= MIN_CREDIBLE_CONFIDENCE
            if contract.confidence_required
            else (
                confidence is None
                or confidence >= MIN_CREDIBLE_CONFIDENCE
            )
        )
        and (
            response_ms is not None
            and response_ms >= MIN_CREDIBLE_RESPONSE_MS
            if contract.response_time_required
            else (
                response_ms is None
                or response_ms >= MIN_CREDIBLE_RESPONSE_MS
            )
        )
    )
    if correct:
        return (
            ResponseClass.CREDIBLE_SUCCESS
            if observable_and_unguided
            else ResponseClass.NONCREDIBLE_SUCCESS
        )
    if not observable_and_unguided:
        return ResponseClass.UNCERTAIN_OR_ABSTAINED
    if selected_misconception_id is not None and (
        not contract.named_error_confidence_required
        or (
            confidence is not None
            and confidence >= MIN_NAMED_ERROR_CONFIDENCE
        )
    ):
        return ResponseClass.CREDIBLE_NAMED_ERROR
    return ResponseClass.CREDIBLE_GENERIC_ERROR


def credible_response_sql(
    *,
    model_expression: str,
    attempt_alias: str = "attempt",
) -> str:
    """Render SQL equivalent to ``classify_response_for_model`` credibility."""

    if (
        not isinstance(model_expression, str)
        or not model_expression.strip()
        or not isinstance(attempt_alias, str)
        or not attempt_alias.isidentifier()
    ):
        raise ValueError("Credibility SQL requires trusted SQL identifiers.")

    groups: dict[tuple[bool, bool], list[str]] = {}
    for model_version, contract in RESPONSE_TELEMETRY_CONTRACTS.items():
        groups.setdefault(
            (
                contract.confidence_required,
                contract.response_time_required,
            ),
            [],
        ).append(model_version)

    cases: list[str] = []
    for (confidence_required, response_time_required), versions in sorted(
        groups.items()
    ):
        quoted = ", ".join(f"'{version}'" for version in sorted(versions))
        confidence_clause = (
            f"{attempt_alias}.confidence IS NOT NULL "
            f"AND {attempt_alias}.confidence >= {MIN_CREDIBLE_CONFIDENCE}"
            if confidence_required
            else (
                f"({attempt_alias}.confidence IS NULL OR "
                f"{attempt_alias}.confidence >= {MIN_CREDIBLE_CONFIDENCE})"
            )
        )
        response_clause = (
            f"{attempt_alias}.response_ms IS NOT NULL "
            f"AND {attempt_alias}.response_ms >= {MIN_CREDIBLE_RESPONSE_MS}"
            if response_time_required
            else (
                f"({attempt_alias}.response_ms IS NULL OR "
                f"{attempt_alias}.response_ms >= {MIN_CREDIBLE_RESPONSE_MS})"
            )
        )
        cases.append(
            f"WHEN {model_expression} IN ({quoted}) "
            f"THEN (({confidence_clause}) AND ({response_clause}))"
        )
    return (
        f"{attempt_alias}.selected_option_id IS NOT NULL "
        f"AND {attempt_alias}.hint_count = 0 "
        "AND CASE "
        + " ".join(cases)
        + " ELSE 0 END"
    )


@dataclass(frozen=True, slots=True)
class ResponseWindow:
    """Authoritative wall window against which submitted active time is checked."""

    elapsed_ms: int
    response_ms: int | None

    @property
    def consistent(self) -> bool:
        return self.response_ms is None or self.response_ms <= self.elapsed_ms


def response_window(
    *,
    selected_at: datetime,
    answered_at: datetime,
    response_ms: int | None,
) -> ResponseWindow:
    """Compare claimed active time with the server-observed selection window.

    The caller validates the scalar bounds.  Timestamps are checked here so
    every command path uses the same fail-closed clock contract.
    """

    for label, timestamp in (
        ("selected_at", selected_at),
        ("answered_at", answered_at),
    ):
        if (
            not isinstance(timestamp, datetime)
            or timestamp.tzinfo is None
            or timestamp.utcoffset() is None
        ):
            raise ValueError(f"{label} must be timezone-aware.")
    if answered_at < selected_at:
        raise ValueError("An answer cannot precede its question selection.")
    elapsed = answered_at - selected_at
    # Avoid binary-float truncation turning an exact millisecond interval into
    # one millisecond less (for example, 3634 ms becoming 3633 ms).  Submission
    # and replay must make the same integer decision on every platform.
    elapsed_microseconds = (
        (elapsed.days * 86_400 + elapsed.seconds) * 1_000_000
        + elapsed.microseconds
    )
    elapsed_ms = elapsed_microseconds // 1_000
    return ResponseWindow(elapsed_ms=elapsed_ms, response_ms=response_ms)
