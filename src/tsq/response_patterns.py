# SPDX-License-Identifier: MPL-2.0

"""Replay-derived, observational diagnostics for response-position habits.

These diagnostics summarize immutable presentation and response evidence.  They
must never update learner state, certify retrieval, or affect question
selection.
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from math import comb
from typing import Any, Iterable, Mapping


DISPLAY_POSITION_DIAGNOSTIC_VERSION = "display-position-shadow-v1"
MIN_POSITION_OBSERVATIONS = 12
POSITION_ANALYSIS_WINDOW = 256
POSITION_FAMILYWISE_ALPHA = Fraction(1, 100)

_EVENT_ENVELOPE_FIELDS = (
    "event_id",
    "stream_id",
    "stream_version",
    "event_type",
    "schema_version",
    "occurred_at",
    "recorded_at",
    "learner_id",
    "session_id",
    "correlation_id",
    "causation_id",
    "idempotency_key",
    "previous_hash",
)


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fraction_payload(value: Fraction) -> dict[str, object]:
    return {
        "value": float(value),
        "exact": f"{value.numerator}/{value.denominator}",
    }


def _exact_poisson_binomial_upper_tail(
    probabilities: Iterable[Fraction],
    observed: int,
) -> Fraction:
    """Return ``P(X >= observed)`` using exact rational arithmetic.

    Equal probabilities use an exact binomial sum with ``O(n)`` terms.  The
    general dynamic program is reserved for mixed option counts and is bounded
    by :data:`POSITION_ANALYSIS_WINDOW` at the public analysis boundary.
    """

    values = tuple(probabilities)
    if observed <= 0:
        return Fraction(1)
    if observed > len(values):
        return Fraction(0)
    if not values:
        return Fraction(0)
    first = values[0]
    if all(probability == first for probability in values[1:]):
        if first == 0:
            return Fraction(0)
        if first == 1:
            return Fraction(1)
        numerator_probability = first.numerator
        denominator_probability = first.denominator
        failure_numerator = denominator_probability - numerator_probability
        numerator = sum(
            comb(len(values), successes)
            * numerator_probability**successes
            * failure_numerator ** (len(values) - successes)
            for successes in range(observed, len(values) + 1)
        )
        return Fraction(numerator, denominator_probability ** len(values))

    distribution = [Fraction(1)]
    for probability in values:
        updated = [Fraction(0)] * (len(distribution) + 1)
        for successes, mass in enumerate(distribution):
            updated[successes] += mass * (1 - probability)
            updated[successes + 1] += mass * probability
        distribution = updated
    if observed >= len(distribution):
        return Fraction(0)
    return sum(distribution[observed:], Fraction(0))


def analyze_position_observations(
    observations: Iterable[tuple[int, int]],
) -> dict[str, Any]:
    """Analyze a validated chronological position stream within a hard window."""

    complete = tuple(observations)
    for selected_position, option_count in complete:
        if (
            type(selected_position) is not int
            or type(option_count) is not int
            or option_count < 2
            or not 0 <= selected_position < option_count
        ):
            raise ValueError("Position observations must fit their option boundary.")
    analyzed = complete[-POSITION_ANALYSIS_WINDOW:]
    window = {
        "total_non_abstained_observations": len(complete),
        "analyzed_non_abstained_observations": len(analyzed),
        "truncated_non_abstained_observations": len(complete) - len(analyzed),
        "maximum_recent_non_abstained_observations": POSITION_ANALYSIS_WINDOW,
        "ordering": (
            "Most recent non-abstained responses in durable answer-time and "
            "attempt-ID order."
        ),
    }
    if len(analyzed) < MIN_POSITION_OBSERVATIONS:
        return {
            "window": window,
            "inference": {
                "status": "inconclusive",
                "reason": "insufficient_non_abstained_observations",
                "position_tests": [],
                "dominant_position": None,
            },
        }

    position_count = max(option_count for _position, option_count in analyzed)
    tests: list[dict[str, Any]] = []
    for position in range(position_count):
        probabilities = tuple(
            Fraction(1, option_count) if position < option_count else Fraction(0)
            for _selected, option_count in analyzed
        )
        count = sum(selected == position for selected, _count in analyzed)
        expected = sum(probabilities, Fraction(0))
        raw = _exact_poisson_binomial_upper_tail(probabilities, count)
        adjusted = min(Fraction(1), raw * position_count)
        calculation = (
            "exact_binomial_equal_probability"
            if len(set(probabilities)) == 1
            else "exact_poisson_binomial_dynamic_program"
        )
        tests.append(
            {
                "display_position": position + 1,
                "selected_count": count,
                "selected_share": count / len(analyzed),
                "expected_count": _fraction_payload(expected),
                "raw_upper_tail_probability": _fraction_payload(raw),
                "bonferroni_adjusted_probability": _fraction_payload(adjusted),
                "calculation": calculation,
                "familywise_signal": (
                    count > expected
                    and adjusted <= POSITION_FAMILYWISE_ALPHA
                ),
            }
        )
    tests.sort(
        key=lambda item: (
            item["bonferroni_adjusted_probability"]["value"],
            -item["selected_count"],
            item["display_position"],
        )
    )
    dominant = tests[0]
    signaled = [item for item in tests if item["familywise_signal"]]
    return {
        "window": window,
        "inference": {
            "status": (
                "position_concentration_signal" if signaled else "no_signal"
            ),
            "reason": (
                "exact_familywise_threshold_crossed"
                if signaled
                else "no_position_crossed_familywise_threshold"
            ),
            "position_tests": sorted(
                tests, key=lambda item: item["display_position"]
            ),
            "dominant_position": dominant,
        },
    }


def _event_payload(
    row: Mapping[str, Any],
    prefix: str,
    *,
    expected_type: str,
    errors: list[str],
) -> dict[str, Any] | None:
    event: dict[str, Any] = {}
    for field in _EVENT_ENVELOPE_FIELDS:
        value = row.get(f"{prefix}_{field}")
        event[field] = value
    payload_json = row.get(f"{prefix}_payload_json")
    metadata_json = row.get(f"{prefix}_metadata_json")
    stored_hash = row.get(f"{prefix}_payload_hash")
    decision_id = row.get("decision_id")
    label = f"decision {decision_id or '<unknown>'} {prefix} event"
    if event["event_id"] is None:
        errors.append(f"{label} is missing")
        return None
    if event["event_type"] != expected_type:
        errors.append(f"{label} has the wrong event type")
        return None
    try:
        payload = json.loads(payload_json)
        metadata = json.loads(metadata_json)
    except (TypeError, ValueError):
        errors.append(f"{label} has malformed JSON")
        return None
    if type(payload) is not dict or type(metadata) is not dict:
        errors.append(f"{label} must contain JSON objects")
        return None
    envelope = {
        **event,
        "payload": payload,
        "metadata": metadata,
    }
    if (
        type(stored_hash) is not str
        or _canonical_hash(envelope) != stored_hash
    ):
        errors.append(f"{label} fails its immutable payload hash")
        return None
    return payload


def _order(
    value: object,
    *,
    label: str,
    errors: list[str],
) -> tuple[str, ...] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            errors.append(f"{label} is malformed JSON")
            return None
    if (
        type(value) is not list
        or len(value) < 2
        or any(type(option_id) is not str or not option_id for option_id in value)
        or len(value) != len(set(value))
    ):
        errors.append(f"{label} is not a unique option-ID list")
        return None
    return tuple(value)


def display_position_shadow(
    rows: Iterable[Mapping[str, Any]],
    selection_event_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Infer a conservative display-position concentration signal.

    The null conditions on each non-abstained response and assumes an
    independent uniform choice among that item's displayed options.  The
    complete immutable history is validated before a bounded recent window is
    analyzed.  Equal option counts use an exact binomial upper tail; mixed
    counts use an exact Poisson-binomial upper tail.  Bonferroni adjustment
    protects the family-wise error rate across every displayed position
    tested.
    """

    materialized = tuple(rows)
    selection_materialized = tuple(selection_event_rows)
    errors: list[str] = []
    selection_by_decision: dict[str, dict[str, Any]] = {}
    for event_row in selection_materialized:
        selection = _event_payload(
            event_row,
            "selection",
            expected_type="QuestionSelected",
            errors=errors,
        )
        if selection is None:
            continue
        decision_id = selection.get("decision_id")
        if type(decision_id) is not str or not decision_id:
            errors.append("a selection event has no decision ID")
            continue
        if decision_id in selection_by_decision:
            errors.append(
                f"decision {decision_id} has more than one selection event"
            )
            continue
        selection_by_decision[decision_id] = selection

    observations: list[tuple[int, int]] = []
    abstentions = 0
    seen_decisions: set[str] = set()
    for row in materialized:
        decision_id = row.get("decision_id")
        if type(decision_id) is not str or not decision_id:
            errors.append("an answered row has no decision ID")
            continue
        if decision_id in seen_decisions:
            errors.append(f"decision {decision_id} appears more than once")
            continue
        seen_decisions.add(decision_id)
        selection = selection_by_decision.get(decision_id)
        if selection is None:
            errors.append(f"decision {decision_id} has no valid selection event")
        response = _event_payload(
            row,
            "response",
            expected_type="ResponseSubmitted",
            errors=errors,
        )
        question_options = _order(
            row.get("question_option_ids_json"),
            label=f"decision {decision_id} question option boundary",
            errors=errors,
        )
        decision_order = _order(
            row.get("decision_option_order_json"),
            label=f"decision {decision_id} projected option order",
            errors=errors,
        )
        attempt_order = _order(
            row.get("attempt_presented_order_json"),
            label=f"decision {decision_id} attempt option order",
            errors=errors,
        )
        selection_order = (
            _order(
                selection.get("option_order"),
                label=f"decision {decision_id} immutable selection option order",
                errors=errors,
            )
            if selection is not None
            else None
        )
        response_order = (
            _order(
                response.get("presented_order"),
                label=f"decision {decision_id} immutable response option order",
                errors=errors,
            )
            if response is not None
            else None
        )
        ordered_boundaries = (
            decision_order,
            attempt_order,
            selection_order,
            response_order,
        )
        if all(boundary is not None for boundary in ordered_boundaries):
            resolved = [
                boundary
                for boundary in ordered_boundaries
                if boundary is not None
            ]
            if any(boundary != resolved[0] for boundary in resolved[1:]):
                errors.append(
                    f"decision {decision_id} has inconsistent option-order boundaries"
                )
            if (
                question_options is not None
                and set(resolved[0]) != set(question_options)
            ):
                errors.append(
                    f"decision {decision_id} option order does not match its question"
                )

        selected = row.get("selected_option_id")
        if response is not None:
            if response.get("decision_id") != decision_id:
                errors.append(
                    f"decision {decision_id} response references another decision"
                )
            if response.get("question_id") != row.get("question_id"):
                errors.append(
                    f"decision {decision_id} response references another question"
                )
            if response.get("selected_option_id") != selected:
                errors.append(
                    f"decision {decision_id} selected option conflicts with its response"
                )
        if selection is not None:
            if selection.get("decision_id") != decision_id:
                errors.append(
                    f"decision {decision_id} selection references another decision"
                )
            if selection.get("question_id") != row.get("question_id"):
                errors.append(
                    f"decision {decision_id} selection references another question"
                )

        if selected is None:
            abstentions += 1
            continue
        if type(selected) is not str:
            errors.append(f"decision {decision_id} selected option is malformed")
            continue
        if response_order is None or selected not in response_order:
            errors.append(
                f"decision {decision_id} selected option is outside its presentation"
            )
            continue
        observations.append((response_order.index(selected), len(response_order)))

    analysis = analyze_position_observations(observations)
    evidence = {
        "answered_observations": len(materialized),
        "selection_event_observations": len(selection_materialized),
        "non_abstained_observations": len(observations),
        "abstentions_excluded": abstentions,
        "boundary_valid": not errors,
        "boundary_errors": errors[:10],
        "source": (
            "Hash-checked QuestionSelected and ResponseSubmitted events, "
            "cross-checked against immutable attempts, projected decisions, "
            "and the release question option set."
        ),
        **analysis["window"],
    }
    contract = {
        "minimum_non_abstained_observations": MIN_POSITION_OBSERVATIONS,
        "maximum_recent_non_abstained_observations": POSITION_ANALYSIS_WINDOW,
        "analysis_window": (
            "Only the most recent eligible non-abstained responses are tested; "
            "the complete immutable session history is still loaded, counted, "
            "hash-checked, and cross-checked before inference."
        ),
        "null_hypothesis": (
            "Conditional on responding, each answer independently chooses "
            "uniformly among that item's displayed options."
        ),
        "tail_test": (
            "Exact binomial upper tail for equal per-item probabilities; "
            "exact Poisson-binomial dynamic program for mixed probabilities."
        ),
        "multiplicity_correction": "Bonferroni across displayed positions",
        "familywise_alpha": float(POSITION_FAMILYWISE_ALPHA),
        "inference_boundary": (
            "A concentration signal is an observational response-habit "
            "hypothesis, not evidence of low knowledge, deception, or causation."
        ),
    }
    base = {
        "diagnostic_version": DISPLAY_POSITION_DIAGNOSTIC_VERSION,
        "observational_only": True,
        "affects_mastery": False,
        "affects_certification": False,
        "affects_selection": False,
        "evidence": evidence,
        "test_contract": contract,
    }
    if errors:
        return {
            **base,
            "inference": {
                "status": "unavailable",
                "reason": "immutable_evidence_boundary_invalid",
                "position_tests": [],
                "dominant_position": None,
            },
        }
    return {
        **base,
        "inference": analysis["inference"],
    }
