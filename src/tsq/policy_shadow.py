# SPDX-License-Identifier: MPL-2.0

"""Replayable, prospective shadow-policy evaluation contracts.

This module deliberately has no live-selection hook.  It can describe and
validate what the frozen ``greedy-policy-score-v1`` challenger would have
selected from the exact safe sampling frontier already produced by the live
policy.  The resulting event and projection are observational only: they do
not select a question, update learner state, or support a causal claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .errors import ValidationError
from .event_contracts import (
    QUESTION_SELECTED_METADATA_FIELDS,
    same_json_value,
)
from .models import CandidateScore


POLICY_SHADOW_EVENT_SCHEMA_VERSION = 1
POLICY_SHADOW_CONTRACT_VERSION = "policy-shadow-v1"
GREEDY_POLICY_VERSION = "greedy-policy-score-v1"
SAFE_FRONTIER_LIMIT = 5
LOGGING_TEMPERATURE = 0.10
SHADOW_REQUIRED_LOGGING_POLICY_VERSIONS = frozenset(
    {"recursive-evidence-graph-v18"}
)

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_SCORE_NUMBER_FIELDS = frozenset(
    {
        "total",
        "predicted_correct",
        "information_gain",
        "learning_fit",
        "concept_need",
        "misconception_value",
        "prerequisite_value",
        "review_value",
        "novelty",
        "kind_fit",
        "continuity",
        "boundary_fit",
        "coverage_diagnostic_information",
    }
)
_SCORE_BOUNDED_FIELDS = _SCORE_NUMBER_FIELDS - {
    "total",
    "coverage_diagnostic_information"
}
_SCORE_INTEGER_FIELDS = frozenset(
    {
        "coverage_raw_exposures",
        "coverage_successful_retrieval_families",
    }
)
_SCORE_FIELDS = _SCORE_NUMBER_FIELDS | _SCORE_INTEGER_FIELDS
_FRONTIER_FIELDS = _SCORE_FIELDS | {
    "question_id",
    "logging_probability",
}

POLICY_SHADOW_PAYLOAD_FIELDS = frozenset(
    {
        "evaluation_id",
        "decision_id",
        "challenger_policy_version",
        "challenger_definition_digest",
        "logging_policy_version",
        "learner_model_version",
        "corpus_release_id",
        "candidate_count",
        "candidate_digest",
        "frontier",
        "frontier_digest",
        "input_digest",
        "output_digest",
        "live_question_id",
        "challenger_question_id",
        "agreement",
        "evaluated_at",
        "shadow_only",
        "selection_applied",
        "mastery_applied",
    }
)
POLICY_SHADOW_METADATA_FIELDS = frozenset(
    {
        "shadow_contract_version",
        "shadow_only",
        "selection_applied",
        "mastery_applied",
    }
)
POLICY_SHADOW_PROJECTION_COLUMNS = (
    "id",
    "event_id",
    "decision_id",
    "challenger_policy_version",
    "challenger_definition_digest",
    "logging_policy_version",
    "learner_model_version",
    "corpus_release_id",
    "candidate_count",
    "candidate_digest",
    "frontier_json",
    "frontier_digest",
    "input_digest",
    "output_digest",
    "live_question_id",
    "challenger_question_id",
    "agreement",
    "evaluated_at",
    "recorded_at",
    "shadow_only",
    "selection_applied",
    "mastery_applied",
)


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"Policy-shadow value is not finite JSON: {exc}") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


GREEDY_POLICY_DEFINITION: dict[str, Any] = {
    "schema_version": 1,
    "policy_version": GREEDY_POLICY_VERSION,
    "input_contract": {
        "candidate_score_terms": "candidate-score-terms-v1",
        "frontier_limit": SAFE_FRONTIER_LIMIT,
        "frontier_order": ["total_desc", "question_id_asc"],
        "logging_probability": {
            "algorithm": "softmax-total-v1",
            "temperature": LOGGING_TEMPERATURE,
        },
    },
    "selection_contract": {
        "algorithm": "greedy-total-v1",
        "score_field": "total",
        "tie_breaker": "question_id_asc",
    },
    "effect_contract": {
        "shadow_only": True,
        "selection_applied": False,
        "mastery_applied": False,
        "causal_claim": False,
    },
}
GREEDY_POLICY_DEFINITION_DIGEST = _digest(GREEDY_POLICY_DEFINITION)


def _strict_object(raw: str, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite JSON number {value}")
        return parsed

    try:
        value = json.loads(
            raw,
            parse_constant=reject_constant,
            parse_float=finite_float,
            object_pairs_hook=reject_duplicate,
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} contains invalid JSON: {exc}") from exc
    if type(value) is not dict:
        raise ValidationError(f"{label} must be a JSON object.")
    return value


def _strict_array(raw: str, label: str) -> list[Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite JSON number {value}")
        return parsed

    try:
        value = json.loads(
            raw,
            parse_constant=reject_constant,
            parse_float=finite_float,
            object_pairs_hook=reject_duplicate,
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} contains invalid JSON: {exc}") from exc
    if type(value) is not list:
        raise ValidationError(f"{label} must be a JSON array.")
    return value


def _require_exact_fields(
    value: Mapping[str, Any], fields: frozenset[str] | set[str], label: str
) -> None:
    missing = sorted(set(fields) - set(value))
    unexpected = sorted(set(value) - set(fields))
    details: list[str] = []
    if missing:
        details.append("missing " + ", ".join(missing))
    if unexpected:
        details.append("unexpected " + ", ".join(unexpected))
    if details:
        raise ValidationError(f"{label} has " + "; ".join(details) + ".")


def _require_id(value: Any, label: str) -> str:
    if type(value) is not str or _ID_PATTERN.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a compact stable identifier.")
    return value


def _require_digest(value: Any, label: str) -> str:
    if type(value) is not str or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValidationError(f"{label} must be an integer of at least {minimum}.")
    return value


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{label} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValidationError(f"{label} must be a finite number.")
    return number


def _canonical_timestamp(value: datetime | str, label: str) -> str:
    try:
        timestamp = (
            value
            if isinstance(value, datetime)
            else datetime.fromisoformat(value)
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(
            f"{label} must be a timezone-aware ISO timestamp."
        ) from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValidationError(f"{label} must be timezone-aware.")
    return timestamp.astimezone(timezone.utc).isoformat()


def _validate_score_terms(
    question_id: Any,
    terms: Mapping[str, Any],
    label: str,
) -> CandidateScore:
    resolved_question_id = _require_id(question_id, f"{label} question_id")
    _require_exact_fields(terms, _SCORE_FIELDS, f"{label} score")
    numbers: dict[str, float] = {}
    for field in _SCORE_NUMBER_FIELDS:
        number = _require_number(terms[field], f"{label} {field}")
        if field in _SCORE_BOUNDED_FIELDS and not 0.0 <= number <= 1.0:
            raise ValidationError(f"{label} {field} must be between zero and one.")
        if field == "coverage_diagnostic_information" and number < 0.0:
            raise ValidationError(
                f"{label} coverage_diagnostic_information must be non-negative."
            )
        numbers[field] = number
    raw_exposures = _require_int(
        terms["coverage_raw_exposures"],
        f"{label} coverage_raw_exposures",
    )
    successful_families = _require_int(
        terms["coverage_successful_retrieval_families"],
        f"{label} coverage_successful_retrieval_families",
    )
    if successful_families > raw_exposures:
        raise ValidationError(
            f"{label} successful retrieval families exceed raw exposures."
        )
    try:
        return CandidateScore(
            question_id=resolved_question_id,
            total=numbers["total"],
            predicted_correct=numbers["predicted_correct"],
            information_gain=numbers["information_gain"],
            learning_fit=numbers["learning_fit"],
            concept_need=numbers["concept_need"],
            misconception_value=numbers["misconception_value"],
            prerequisite_value=numbers["prerequisite_value"],
            review_value=numbers["review_value"],
            novelty=numbers["novelty"],
            kind_fit=numbers["kind_fit"],
            continuity=numbers["continuity"],
            boundary_fit=numbers["boundary_fit"],
            coverage_raw_exposures=raw_exposures,
            coverage_diagnostic_information=numbers[
                "coverage_diagnostic_information"
            ],
            coverage_successful_retrieval_families=successful_families,
        )
    except ValueError as exc:
        raise ValidationError(f"{label} is not a valid CandidateScore: {exc}") from exc


def policy_shadow_logging_probabilities(
    scores: Sequence[CandidateScore],
) -> tuple[float, ...]:
    """Return the frozen live top-frontier softmax probabilities."""

    if not scores:
        raise ValidationError("Policy-shadow frontier must not be empty.")
    totals = [
        _require_number(score.total, f"Frontier candidate {index} total")
        for index, score in enumerate(scores)
    ]
    peak = max(totals)
    weights = [
        math.exp((total - peak) / LOGGING_TEMPERATURE) for total in totals
    ]
    total_weight = sum(weights)
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValidationError("Policy-shadow logging weights are invalid.")
    probabilities = tuple(weight / total_weight for weight in weights)
    if any(
        not math.isfinite(probability) or probability <= 0.0
        for probability in probabilities
    ):
        raise ValidationError(
            "Policy-shadow logging probabilities lost full support through "
            "numerical underflow."
        )
    return probabilities


def _canonical_frontier(
    frontier: Sequence[tuple[CandidateScore, float]],
    *,
    candidate_count: int,
    live_question_id: str,
) -> tuple[list[dict[str, Any]], tuple[CandidateScore, ...]]:
    if type(candidate_count) is not int or candidate_count < 1:
        raise ValidationError("candidate_count must be a positive integer.")
    required_size = min(candidate_count, SAFE_FRONTIER_LIMIT)
    if len(frontier) != required_size:
        raise ValidationError(
            "Policy-shadow frontier must contain exactly the safe top-k "
            f"frontier ({required_size} candidates)."
        )
    scores: list[CandidateScore] = []
    supplied_probabilities: list[float] = []
    for index, item in enumerate(frontier):
        if type(item) not in {tuple, list} or len(item) != 2:
            raise ValidationError(
                f"Frontier entry {index} must be (CandidateScore, probability)."
            )
        score, probability = item
        if not isinstance(score, CandidateScore):
            raise ValidationError(
                f"Frontier entry {index} must contain a CandidateScore."
            )
        validated = _validate_score_terms(
            score.question_id,
            score.terms(),
            f"Frontier entry {index}",
        )
        supplied = _require_number(
            probability, f"Frontier entry {index} logging_probability"
        )
        if not 0.0 < supplied <= 1.0:
            raise ValidationError(
                f"Frontier entry {index} logging_probability must be in (0, 1]."
            )
        scores.append(validated)
        supplied_probabilities.append(supplied)
    ids = [score.question_id for score in scores]
    if len(ids) != len(set(ids)):
        raise ValidationError("Policy-shadow frontier question IDs must be unique.")
    expected_order = sorted(scores, key=lambda score: (-score.total, score.question_id))
    if ids != [score.question_id for score in expected_order]:
        raise ValidationError(
            "Policy-shadow frontier must be ordered by total descending and "
            "question ID ascending."
        )
    if live_question_id not in ids:
        raise ValidationError(
            "The live selected question must appear in the safe top-k frontier."
        )
    expected_probabilities = policy_shadow_logging_probabilities(scores)
    for index, (supplied, expected) in enumerate(
        zip(supplied_probabilities, expected_probabilities, strict=True)
    ):
        if supplied != expected:
            raise ValidationError(
                f"Frontier entry {index} logging_probability does not match "
                "the frozen softmax contract."
            )
    canonical = [
        {
            "question_id": score.question_id,
            **score.terms(),
            "logging_probability": probability,
        }
        for score, probability in zip(
            scores, expected_probabilities, strict=True
        )
    ]
    return canonical, tuple(scores)


@dataclass(frozen=True, slots=True)
class BuiltPolicyShadowEvaluation:
    """Canonical event material produced before the learner responds."""

    payload: dict[str, Any]
    metadata: dict[str, Any]

    @property
    def evaluation_id(self) -> str:
        return self.payload["evaluation_id"]

    def projection_row(
        self, *, event_id: str, recorded_at: datetime | str
    ) -> dict[str, Any]:
        """Return the exact mutable projection row for an appended event."""

        _require_id(event_id, "Policy-shadow event ID")
        return {
            "id": self.payload["evaluation_id"],
            "event_id": event_id,
            "decision_id": self.payload["decision_id"],
            "challenger_policy_version": self.payload[
                "challenger_policy_version"
            ],
            "challenger_definition_digest": self.payload[
                "challenger_definition_digest"
            ],
            "logging_policy_version": self.payload["logging_policy_version"],
            "learner_model_version": self.payload["learner_model_version"],
            "corpus_release_id": self.payload["corpus_release_id"],
            "candidate_count": self.payload["candidate_count"],
            "candidate_digest": self.payload["candidate_digest"],
            "frontier_json": _canonical_json(self.payload["frontier"]),
            "frontier_digest": self.payload["frontier_digest"],
            "input_digest": self.payload["input_digest"],
            "output_digest": self.payload["output_digest"],
            "live_question_id": self.payload["live_question_id"],
            "challenger_question_id": self.payload["challenger_question_id"],
            "agreement": int(self.payload["agreement"]),
            "evaluated_at": self.payload["evaluated_at"],
            "recorded_at": _canonical_timestamp(
                recorded_at, "Policy-shadow recording time"
            ),
            "shadow_only": 1,
            "selection_applied": 0,
            "mastery_applied": 0,
        }


def build_policy_shadow_evaluation(
    *,
    decision_id: str,
    logging_policy_version: str,
    learner_model_version: str,
    corpus_release_id: str,
    candidate_count: int,
    candidate_digest: str,
    frontier: Sequence[tuple[CandidateScore, float]],
    live_question_id: str,
    evaluated_at: datetime | str,
) -> BuiltPolicyShadowEvaluation:
    """Build one deterministic prospective challenger evaluation.

    ``frontier`` must be the complete safe sampling frontier in live-policy
    order.  Supplied propensities are verified against the frozen logging
    softmax; they are never trusted merely because they sum to one.
    """

    resolved_decision_id = _require_id(decision_id, "decision_id")
    resolved_logging_policy = _require_id(
        logging_policy_version, "logging_policy_version"
    )
    resolved_model = _require_id(
        learner_model_version, "learner_model_version"
    )
    resolved_release = _require_id(corpus_release_id, "corpus_release_id")
    resolved_live_question = _require_id(
        live_question_id, "live_question_id"
    )
    resolved_candidate_digest = _require_digest(
        candidate_digest, "candidate_digest"
    )
    resolved_candidate_count = _require_int(
        candidate_count, "candidate_count", minimum=1
    )
    evaluated_timestamp = _canonical_timestamp(evaluated_at, "evaluated_at")
    canonical_frontier, scores = _canonical_frontier(
        frontier,
        candidate_count=resolved_candidate_count,
        live_question_id=resolved_live_question,
    )
    challenger_question_id = min(
        scores, key=lambda score: (-score.total, score.question_id)
    ).question_id
    agreement = challenger_question_id == resolved_live_question
    frontier_digest = _digest(canonical_frontier)
    input_terms = {
        "shadow_contract_version": POLICY_SHADOW_CONTRACT_VERSION,
        "decision_id": resolved_decision_id,
        "challenger_policy_version": GREEDY_POLICY_VERSION,
        "challenger_definition_digest": GREEDY_POLICY_DEFINITION_DIGEST,
        "logging_policy_version": resolved_logging_policy,
        "learner_model_version": resolved_model,
        "corpus_release_id": resolved_release,
        "candidate_count": resolved_candidate_count,
        "candidate_digest": resolved_candidate_digest,
        "frontier": canonical_frontier,
        "frontier_digest": frontier_digest,
        "live_question_id": resolved_live_question,
        "evaluated_at": evaluated_timestamp,
    }
    input_digest = _digest(input_terms)
    output_terms = {
        "input_digest": input_digest,
        "challenger_question_id": challenger_question_id,
        "agreement": agreement,
        "shadow_only": True,
        "selection_applied": False,
        "mastery_applied": False,
        "causal_claim": False,
    }
    output_digest = _digest(output_terms)
    evaluation_id = f"pse_{output_digest[:24]}"
    payload = {
        "evaluation_id": evaluation_id,
        "decision_id": resolved_decision_id,
        "challenger_policy_version": GREEDY_POLICY_VERSION,
        "challenger_definition_digest": GREEDY_POLICY_DEFINITION_DIGEST,
        "logging_policy_version": resolved_logging_policy,
        "learner_model_version": resolved_model,
        "corpus_release_id": resolved_release,
        "candidate_count": resolved_candidate_count,
        "candidate_digest": resolved_candidate_digest,
        "frontier": canonical_frontier,
        "frontier_digest": frontier_digest,
        "input_digest": input_digest,
        "output_digest": output_digest,
        "live_question_id": resolved_live_question,
        "challenger_question_id": challenger_question_id,
        "agreement": agreement,
        "evaluated_at": evaluated_timestamp,
        "shadow_only": True,
        "selection_applied": False,
        "mastery_applied": False,
    }
    metadata = {
        "shadow_contract_version": POLICY_SHADOW_CONTRACT_VERSION,
        "shadow_only": True,
        "selection_applied": False,
        "mastery_applied": False,
    }
    return BuiltPolicyShadowEvaluation(payload=payload, metadata=metadata)


def _frontier_from_payload(value: Any, label: str) -> list[tuple[CandidateScore, float]]:
    if type(value) is not list:
        raise ValidationError(f"{label} must be an array.")
    frontier: list[tuple[CandidateScore, float]] = []
    for index, entry in enumerate(value):
        entry_label = f"{label} entry {index}"
        if type(entry) is not dict:
            raise ValidationError(f"{entry_label} must be an object.")
        _require_exact_fields(entry, _FRONTIER_FIELDS, entry_label)
        score = _validate_score_terms(
            entry["question_id"],
            {field: entry[field] for field in _SCORE_FIELDS},
            entry_label,
        )
        probability = _require_number(
            entry["logging_probability"],
            f"{entry_label} logging_probability",
        )
        frontier.append((score, probability))
    return frontier


def _row_dict(row: sqlite3.Row | Mapping[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _projection_table_exists(connection: sqlite3.Connection) -> bool:
    return (
        connection.execute(
            """SELECT 1 FROM sqlite_master
               WHERE type='table' AND name='policy_shadow_evaluations'"""
        ).fetchone()
        is not None
    )


def _projection_from_event(
    connection: sqlite3.Connection,
    event: sqlite3.Row,
) -> tuple[dict[str, Any], dict[str, Any]]:
    label = f"PolicyShadowEvaluated event {event['event_id']}"
    if event["event_type"] != "PolicyShadowEvaluated":
        raise ValidationError(f"{label} has an unexpected event type.")
    if event["schema_version"] != POLICY_SHADOW_EVENT_SCHEMA_VERSION:
        raise ValidationError(
            f"{label} uses unsupported schema {event['schema_version']}."
        )
    payload = _strict_object(event["payload_json"], f"{label} payload")
    metadata = _strict_object(event["metadata_json"], f"{label} metadata")
    _require_exact_fields(payload, POLICY_SHADOW_PAYLOAD_FIELDS, f"{label} payload")
    _require_exact_fields(
        metadata, POLICY_SHADOW_METADATA_FIELDS, f"{label} metadata"
    )
    expected_metadata = {
        "shadow_contract_version": POLICY_SHADOW_CONTRACT_VERSION,
        "shadow_only": True,
        "selection_applied": False,
        "mastery_applied": False,
    }
    if not same_json_value(metadata, expected_metadata):
        raise ValidationError(f"{label} metadata violates the shadow boundary.")
    if (
        payload["shadow_only"] is not True
        or payload["selection_applied"] is not False
        or payload["mastery_applied"] is not False
    ):
        raise ValidationError(f"{label} payload violates the shadow boundary.")

    frontier = _frontier_from_payload(payload["frontier"], f"{label} frontier")
    rebuilt = build_policy_shadow_evaluation(
        decision_id=payload["decision_id"],
        logging_policy_version=payload["logging_policy_version"],
        learner_model_version=payload["learner_model_version"],
        corpus_release_id=payload["corpus_release_id"],
        candidate_count=payload["candidate_count"],
        candidate_digest=payload["candidate_digest"],
        frontier=frontier,
        live_question_id=payload["live_question_id"],
        evaluated_at=payload["evaluated_at"],
    )
    if not same_json_value(payload, rebuilt.payload):
        raise ValidationError(
            f"{label} does not match deterministic challenger derivation."
        )
    if event["correlation_id"] != payload["decision_id"]:
        raise ValidationError(f"{label} correlation ID is not its decision ID.")
    expected_idempotency_key = (
        "policy-shadow:v1:"
        + payload["decision_id"]
        + ":"
        + payload["challenger_definition_digest"]
    )
    if event["idempotency_key"] != expected_idempotency_key:
        raise ValidationError(f"{label} has an invalid idempotency boundary.")
    if event["occurred_at"] != payload["evaluated_at"]:
        raise ValidationError(f"{label} occurrence time is not its evaluation time.")

    decision = connection.execute(
        """SELECT decision.*, session.learner_id AS session_learner_id
           FROM decisions decision
           JOIN sessions session ON session.id=decision.session_id
           WHERE decision.id=?""",
        (payload["decision_id"],),
    ).fetchone()
    if decision is None:
        raise ValidationError(f"{label} cites an unknown decision.")
    if (
        event["learner_id"] != decision["learner_id"]
        or event["learner_id"] != decision["session_learner_id"]
        or event["session_id"] != decision["session_id"]
        or event["stream_id"] != f"learner:{decision['learner_id']}"
    ):
        raise ValidationError(f"{label} does not match its decision envelope.")
    if (
        payload["logging_policy_version"] != decision["policy_version"]
        or payload["corpus_release_id"] != decision["corpus_release_id"]
        or payload["candidate_count"] != decision["candidate_count"]
        or payload["candidate_digest"] != decision["candidate_digest"]
        or payload["live_question_id"] != decision["question_id"]
    ):
        raise ValidationError(f"{label} does not match its live decision.")

    selection = connection.execute(
        "SELECT * FROM events WHERE event_id=?",
        (event["causation_id"],),
    ).fetchone()
    if selection is None or selection["event_type"] != "QuestionSelected":
        raise ValidationError(f"{label} lacks its QuestionSelected cause.")
    selection_payload = _strict_object(
        selection["payload_json"],
        f"QuestionSelected event {selection['event_id']} payload",
    )
    selection_metadata = _strict_object(
        selection["metadata_json"],
        f"QuestionSelected event {selection['event_id']} metadata",
    )
    _require_exact_fields(
        selection_metadata,
        QUESTION_SELECTED_METADATA_FIELDS,
        f"QuestionSelected event {selection['event_id']} metadata",
    )
    if (
        selection_payload.get("decision_id") != decision["id"]
        or selection_payload.get("question_id") != decision["question_id"]
        or selection_payload.get("candidate_count") != decision["candidate_count"]
        or selection_payload.get("candidate_digest") != decision["candidate_digest"]
        or selection_payload.get("corpus_release_id")
        != decision["corpus_release_id"]
        or selection_metadata["policy_version"] != decision["policy_version"]
        or selection_metadata["learner_model_version"]
        != payload["learner_model_version"]
        or selection_metadata["corpus_release_id"]
        != decision["corpus_release_id"]
    ):
        raise ValidationError(f"{label} does not match its selection boundary.")
    if (
        selection["stream_id"] != event["stream_id"]
        or selection["learner_id"] != event["learner_id"]
        or selection["session_id"] != event["session_id"]
        or selection["stream_version"] + 1 != event["stream_version"]
        or selection["occurred_at"] != event["occurred_at"]
    ):
        raise ValidationError(f"{label} was not recorded prospectively at selection.")

    stored_frontier = _strict_array(
        decision["top_candidates_json"],
        f"Decision {decision['id']} top candidates",
    )
    required_size = min(decision["candidate_count"], SAFE_FRONTIER_LIMIT)
    if len(stored_frontier) < required_size:
        raise ValidationError(
            f"{label} live decision lacks its complete safe frontier."
        )
    expected_frontier: list[dict[str, Any]] = []
    expected_scores: list[CandidateScore] = []
    for index, stored in enumerate(stored_frontier[:required_size]):
        item_label = f"Decision {decision['id']} candidate {index}"
        if type(stored) is not dict:
            raise ValidationError(f"{item_label} must be an object.")
        _require_exact_fields(
            stored,
            _SCORE_FIELDS | {"question_id"},
            item_label,
        )
        score = _validate_score_terms(
            stored["question_id"],
            {field: stored[field] for field in _SCORE_FIELDS},
            item_label,
        )
        expected_scores.append(score)
    expected_probabilities = policy_shadow_logging_probabilities(expected_scores)
    for score, probability in zip(
        expected_scores, expected_probabilities, strict=True
    ):
        expected_frontier.append(
            {
                "question_id": score.question_id,
                **score.terms(),
                "logging_probability": probability,
            }
        )
    if not same_json_value(payload["frontier"], expected_frontier):
        raise ValidationError(f"{label} frontier differs from the live safe frontier.")
    live_indexes = [
        index
        for index, score in enumerate(expected_scores)
        if score.question_id == payload["live_question_id"]
    ]
    if len(live_indexes) != 1:
        raise ValidationError(
            f"{label} live question is not unique in the live safe frontier."
        )
    live_index = live_indexes[0]
    expected_live_score = expected_scores[live_index]
    expected_propensity = expected_probabilities[live_index]
    decision_selected_score = _strict_object(
        decision["selected_score_json"],
        f"Decision {decision['id']} selected score",
    )
    _require_exact_fields(
        decision_selected_score,
        _SCORE_FIELDS,
        f"Decision {decision['id']} selected score",
    )
    _validate_score_terms(
        payload["live_question_id"],
        decision_selected_score,
        f"Decision {decision['id']} selected score",
    )
    if not same_json_value(
        decision_selected_score, expected_live_score.terms()
    ):
        raise ValidationError(
            f"{label} selected score differs from the immutable safe frontier."
        )
    decision_propensity = _require_number(
        decision["propensity"],
        f"Decision {decision['id']} propensity",
    )
    if (
        not 0.0 < decision_propensity <= 1.0
        or decision_propensity != expected_propensity
    ):
        raise ValidationError(
            f"{label} propensity differs from the immutable safe frontier."
        )
    selection_score = selection_payload.get("score")
    if type(selection_score) is not dict:
        raise ValidationError(f"{label} selection score must be an object.")
    _require_exact_fields(
        selection_score,
        _SCORE_FIELDS,
        f"{label} selection score",
    )
    _validate_score_terms(
        payload["live_question_id"],
        selection_score,
        f"{label} selection score",
    )
    if not same_json_value(selection_score, expected_live_score.terms()):
        raise ValidationError(
            f"{label} selection score differs from the immutable safe frontier."
        )
    selection_propensity = _require_number(
        selection_payload.get("propensity"),
        f"{label} selection propensity",
    )
    if (
        not 0.0 < selection_propensity <= 1.0
        or selection_propensity != expected_propensity
    ):
        raise ValidationError(
            f"{label} selection propensity differs from the immutable safe "
            "frontier."
        )

    row = rebuilt.projection_row(
        event_id=event["event_id"],
        recorded_at=event["recorded_at"],
    )
    checkpoint = {
        "event_id": event["event_id"],
        "evaluation_id": payload["evaluation_id"],
        "decision_id": payload["decision_id"],
        "live_question_id": payload["live_question_id"],
        "challenger_question_id": payload["challenger_question_id"],
        "agreement": payload["agreement"],
    }
    return row, checkpoint


def policy_shadow_projection_snapshot(
    connection: sqlite3.Connection,
) -> dict[str, list[dict[str, Any]]]:
    """Return the exact mutable shadow projection in stable order."""

    if not _projection_table_exists(connection):
        return {"evaluations": []}
    rows = connection.execute(
        "SELECT * FROM policy_shadow_evaluations ORDER BY id"
    ).fetchall()
    return {"evaluations": [_row_dict(row) for row in rows]}


def derive_policy_shadow_projections(
    connection: sqlite3.Connection,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Derive policy-shadow rows exclusively from immutable events."""

    rows: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    evaluation_ids: set[str] = set()
    decision_ids: set[str] = set()
    for event in connection.execute(
        """SELECT * FROM events
           WHERE event_type='PolicyShadowEvaluated'
           ORDER BY stream_id, stream_version"""
    ).fetchall():
        row, checkpoint = _projection_from_event(connection, event)
        if row["id"] in evaluation_ids:
            raise ValidationError(
                f"Policy-shadow evaluation ID {row['id']} is repeated."
            )
        if row["decision_id"] in decision_ids:
            raise ValidationError(
                f"Decision {row['decision_id']} has multiple shadow evaluations."
            )
        evaluation_ids.add(row["id"])
        decision_ids.add(row["decision_id"])
        rows.append(row)
        checkpoints.append(checkpoint)
    rows.sort(key=lambda row: row["id"])
    checkpoints.sort(key=lambda row: (row["decision_id"], row["evaluation_id"]))
    return {"evaluations": rows}, checkpoints


def rebuild_policy_shadow_projections(
    connection: sqlite3.Connection,
) -> dict[str, Any]:
    """Replace the mutable shadow projection from events on a database copy."""

    snapshot, checkpoints = derive_policy_shadow_projections(connection)
    if not _projection_table_exists(connection):
        if snapshot["evaluations"]:
            raise ValidationError(
                "Policy-shadow events exist without a projection table."
            )
        return {
            "snapshot": {"evaluations": []},
            "checkpoints": checkpoints,
            "evaluation_count": 0,
            "projection_hash": _digest({"evaluations": []}),
        }
    triggers = connection.execute(
        """SELECT name, sql FROM sqlite_master
           WHERE type='trigger' AND tbl_name='policy_shadow_evaluations'
           ORDER BY name"""
    ).fetchall()
    for trigger in triggers:
        escaped = trigger["name"].replace('"', '""')
        connection.execute(f'DROP TRIGGER "{escaped}"')
    connection.execute("DELETE FROM policy_shadow_evaluations")
    placeholders = ", ".join(f":{column}" for column in POLICY_SHADOW_PROJECTION_COLUMNS)
    connection.executemany(
        "INSERT INTO policy_shadow_evaluations("
        + ", ".join(POLICY_SHADOW_PROJECTION_COLUMNS)
        + ") VALUES ("
        + placeholders
        + ")",
        snapshot["evaluations"],
    )
    # Import lazily so :mod:`tsq.store` may register the integrity function
    # without creating a module-import cycle.
    from .store import Database

    installer = getattr(Database, "_install_policy_shadow_triggers", None)
    if installer is not None:
        installer(connection)
    elif triggers:
        for trigger in triggers:
            if trigger["sql"] is None:
                raise ValidationError(
                    f"Policy-shadow trigger {trigger['name']} has no SQL."
                )
            connection.execute(trigger["sql"])
    rebuilt = policy_shadow_projection_snapshot(connection)
    return {
        "snapshot": rebuilt,
        "checkpoints": checkpoints,
        "evaluation_count": len(snapshot["evaluations"]),
        "projection_hash": _digest(rebuilt),
    }


def policy_shadow_integrity_errors(
    connection: sqlite3.Connection,
) -> list[str]:
    """Return fail-closed event/projection errors without mutating state."""

    event_count = connection.execute(
        """SELECT COUNT(*) AS n FROM events
           WHERE event_type='PolicyShadowEvaluated'"""
    ).fetchone()["n"]
    if not _projection_table_exists(connection):
        return (
            ["policy shadow: events exist without policy_shadow_evaluations"]
            if event_count
            else []
        )
    try:
        expected, _ = derive_policy_shadow_projections(connection)
    except (sqlite3.DatabaseError, TypeError, ValueError, ValidationError) as exc:
        return [f"policy shadow: {exc}"]
    actual = policy_shadow_projection_snapshot(connection)
    expected_by_id = {
        row["id"]: row for row in expected["evaluations"]
    }
    actual_by_id = {row["id"]: row for row in actual["evaluations"]}
    errors: list[str] = []
    for evaluation_id in sorted(expected_by_id.keys() - actual_by_id.keys()):
        errors.append(
            f"policy shadow evaluation {evaluation_id}: missing projection row"
        )
    for evaluation_id in sorted(actual_by_id.keys() - expected_by_id.keys()):
        errors.append(
            f"policy shadow evaluation {evaluation_id}: projection has no event"
        )
    for evaluation_id in sorted(expected_by_id.keys() & actual_by_id.keys()):
        if not same_json_value(
            actual_by_id[evaluation_id], expected_by_id[evaluation_id]
        ):
            errors.append(
                f"policy shadow evaluation {evaluation_id}: "
                "projection differs from deterministic event derivation"
            )
    missing_event_decisions = connection.execute(
        """SELECT decision.id
           FROM decisions decision
           WHERE decision.policy_version IN (
               SELECT value FROM json_each(?)
           )
             AND NOT EXISTS (
                 SELECT 1
                 FROM events event
                 WHERE event.event_type='PolicyShadowEvaluated'
                   AND json_valid(event.payload_json)
                   AND json_extract(
                       event.payload_json, '$.decision_id'
                   )=decision.id
             )
           ORDER BY decision.id""",
        (_canonical_json(sorted(SHADOW_REQUIRED_LOGGING_POLICY_VERSIONS)),),
    ).fetchall()
    errors.extend(
        "policy shadow decision "
        + row["id"]
        + ": missing prospective evaluation event"
        for row in missing_event_decisions
    )
    return errors


__all__ = [
    "BuiltPolicyShadowEvaluation",
    "GREEDY_POLICY_DEFINITION",
    "GREEDY_POLICY_DEFINITION_DIGEST",
    "GREEDY_POLICY_VERSION",
    "LOGGING_TEMPERATURE",
    "POLICY_SHADOW_CONTRACT_VERSION",
    "POLICY_SHADOW_EVENT_SCHEMA_VERSION",
    "POLICY_SHADOW_METADATA_FIELDS",
    "POLICY_SHADOW_PAYLOAD_FIELDS",
    "POLICY_SHADOW_PROJECTION_COLUMNS",
    "SAFE_FRONTIER_LIMIT",
    "SHADOW_REQUIRED_LOGGING_POLICY_VERSIONS",
    "build_policy_shadow_evaluation",
    "derive_policy_shadow_projections",
    "policy_shadow_integrity_errors",
    "policy_shadow_logging_probabilities",
    "policy_shadow_projection_snapshot",
    "rebuild_policy_shadow_projections",
]
