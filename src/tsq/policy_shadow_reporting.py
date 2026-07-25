# SPDX-License-Identifier: MPL-2.0

"""Read-only diagnostics for logged and prospective adaptive-policy choices.

The estimands in this module are intentionally narrow.  Historical
off-policy estimates compare a one-step uniform policy over the *same* safe
frontier visited by the logging policy.  Prospective challenger outcomes are
reported only when the challenger selected the action that was actually
served.  Neither analysis reconstructs an unobserved learner trajectory.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import NotFoundError, ValidationError
from .event_contracts import (
    QUESTION_SELECTED_BASE_FIELDS,
    QUESTION_SELECTED_METADATA_FIELDS,
    QUESTION_SELECTED_OBJECTIVE_FIELDS,
    RESPONSE_FIELDS,
    RESPONSE_METADATA_FIELDS,
    RESPONSE_METADATA_FIELDS_WITH_MISCONCEPTION_ALGORITHM,
    same_json_value,
)
from .inference import (
    MISCONCEPTION_ALGORITHM_METADATA_KEY,
    MISCONCEPTION_ALGORITHM_VERSION,
    ResponseClass,
    classify_response_for_model,
)
from .models import CandidateScore
from .policy_shadow import (
    POLICY_SHADOW_CONTRACT_VERSION,
    SAFE_FRONTIER_LIMIT,
    policy_shadow_integrity_errors,
    policy_shadow_logging_probabilities,
    policy_shadow_projection_snapshot,
)
from .store import Database
from .versions import SUPPORTED_MODEL_VERSIONS, question_selected_schema_for


POLICY_SHADOW_REPORT_VERSION = "policy-shadow-report-v1"
UNIFORM_SAFE_FRONTIER_POLICY_VERSION = "uniform-safe-frontier-v1"
SUPPORTED_LOGGING_POLICY_VERSIONS = (
    "recursive-evidence-graph-v17",
    "recursive-evidence-graph-v18",
)
CALIBRATION_BIN_COUNT = 10
LOG_LOSS_CLIP = 1e-9
MIN_EFFECTIVE_SAMPLE_SIZE = 30.0
MIN_EFFECTIVE_SAMPLE_RATIO = 0.10

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
_CANDIDATE_FIELDS = _SCORE_FIELDS | {"question_id"}


@dataclass(frozen=True, slots=True)
class _DecisionBoundary:
    decision_id: str
    session_id: str
    learner_id: str
    prediction: float
    propensity: float
    frontier_size: int
    model_version: str


@dataclass(frozen=True, slots=True)
class _Observation:
    decision_id: str
    prediction: float
    correct: int
    credible_retrieval: int
    propensity: float
    target_probability: float

    @property
    def importance_weight(self) -> float:
        return self.target_probability / self.propensity


def _reject_duplicate_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number {value}")
    return parsed


def _strict_json(raw: Any, label: str, expected_type: type) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    if type(raw) is not str:
        raise ValidationError(f"{label} must be serialized JSON.")
    try:
        value = json.loads(
            raw,
            parse_constant=reject_constant,
            parse_float=_finite_json_float,
            object_pairs_hook=_reject_duplicate_object,
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} contains invalid JSON: {exc}") from exc
    if type(value) is not expected_type:
        expected = "object" if expected_type is dict else "array"
        raise ValidationError(f"{label} must be a JSON {expected}.")
    return value


def _strict_object(raw: Any, label: str) -> dict[str, Any]:
    return _strict_json(raw, label, dict)


def _strict_array(raw: Any, label: str) -> list[Any]:
    return _strict_json(raw, label, list)


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str] | set[str],
    label: str,
) -> None:
    missing = set(expected) - set(value)
    unknown = set(value) - set(expected)
    if not missing and not unknown:
        return
    details: list[str] = []
    if missing:
        details.append("missing " + ", ".join(sorted(missing)))
    if unknown:
        details.append("unknown " + ", ".join(sorted(unknown)))
    raise ValidationError(
        f"{label} has incompatible fields ({'; '.join(details)})."
    )


def _require_nonblank(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValidationError(f"{label} must be a non-blank string.")
    return value


def _require_digest(value: Any, label: str) -> str:
    if type(value) is not str or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValidationError(
            f"{label} must be an integer of at least {minimum}."
        )
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


def _require_probability(value: Any, label: str, *, positive: bool) -> float:
    number = _require_number(value, label)
    lower_valid = number > 0.0 if positive else number >= 0.0
    if not lower_valid or number > 1.0:
        interval = "(0, 1]" if positive else "[0, 1]"
        raise ValidationError(f"{label} must be in {interval}.")
    return number


def _validate_score(
    value: Any,
    *,
    question_id: str,
    label: str,
) -> CandidateScore:
    if type(value) is not dict:
        raise ValidationError(f"{label} must be an object.")
    _require_exact_fields(value, _SCORE_FIELDS, label)
    numbers: dict[str, float] = {}
    for field in _SCORE_NUMBER_FIELDS:
        number = _require_number(value[field], f"{label} {field}")
        if field in _SCORE_BOUNDED_FIELDS and not 0.0 <= number <= 1.0:
            raise ValidationError(
                f"{label} {field} must be between zero and one."
            )
        if field == "coverage_diagnostic_information" and number < 0.0:
            raise ValidationError(
                f"{label} coverage_diagnostic_information must be non-negative."
            )
        numbers[field] = number
    raw_exposures = _require_int(
        value["coverage_raw_exposures"],
        f"{label} coverage_raw_exposures",
    )
    successful_families = _require_int(
        value["coverage_successful_retrieval_families"],
        f"{label} coverage_successful_retrieval_families",
    )
    if successful_families > raw_exposures:
        raise ValidationError(
            f"{label} successful retrieval families exceed raw exposures."
        )
    try:
        return CandidateScore(
            question_id=question_id,
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
        raise ValidationError(f"{label} is invalid: {exc}") from exc


def _validate_frontier(
    decision: sqlite3.Row,
) -> tuple[tuple[CandidateScore, ...], CandidateScore, float]:
    decision_id = decision["id"]
    candidate_count = _require_int(
        decision["candidate_count"],
        f"Decision {decision_id} candidate_count",
        minimum=1,
    )
    _require_digest(
        decision["candidate_digest"],
        f"Decision {decision_id} candidate_digest",
    )
    stored = _strict_array(
        decision["top_candidates_json"],
        f"Decision {decision_id} top candidates",
    )
    expected_count = min(candidate_count, 10)
    if len(stored) != expected_count:
        raise ValidationError(
            f"Decision {decision_id} must retain exactly {expected_count} "
            "audited candidates."
        )
    scores: list[CandidateScore] = []
    for index, entry in enumerate(stored):
        label = f"Decision {decision_id} candidate {index}"
        if type(entry) is not dict:
            raise ValidationError(f"{label} must be an object.")
        _require_exact_fields(entry, _CANDIDATE_FIELDS, label)
        question_id = _require_nonblank(
            entry["question_id"], f"{label} question_id"
        )
        score = _validate_score(
            {field: entry[field] for field in _SCORE_FIELDS},
            question_id=question_id,
            label=f"{label} score",
        )
        scores.append(score)
    ids = [score.question_id for score in scores]
    if len(ids) != len(set(ids)):
        raise ValidationError(
            f"Decision {decision_id} has duplicate audited candidates."
        )
    expected_order = sorted(
        scores, key=lambda score: (-score.total, score.question_id)
    )
    if ids != [score.question_id for score in expected_order]:
        raise ValidationError(
            f"Decision {decision_id} candidates violate the logged ordering."
        )
    frontier_size = min(SAFE_FRONTIER_LIMIT, candidate_count)
    frontier = tuple(scores[:frontier_size])
    live_question_id = _require_nonblank(
        decision["question_id"], f"Decision {decision_id} question_id"
    )
    live_indexes = [
        index
        for index, score in enumerate(frontier)
        if score.question_id == live_question_id
    ]
    if len(live_indexes) != 1:
        raise ValidationError(
            f"Decision {decision_id} selected question is not uniquely present "
            "in its safe frontier."
        )
    selected = frontier[live_indexes[0]]
    selected_score = _strict_object(
        decision["selected_score_json"],
        f"Decision {decision_id} selected score",
    )
    _validate_score(
        selected_score,
        question_id=live_question_id,
        label=f"Decision {decision_id} selected score",
    )
    if not same_json_value(selected_score, selected.terms()):
        raise ValidationError(
            f"Decision {decision_id} selected score differs from its frontier."
        )
    probabilities = policy_shadow_logging_probabilities(frontier)
    stored_propensity = _require_probability(
        decision["propensity"],
        f"Decision {decision_id} propensity",
        positive=True,
    )
    expected_propensity = probabilities[live_indexes[0]]
    if not math.isclose(
        stored_propensity,
        expected_propensity,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise ValidationError(
            f"Decision {decision_id} propensity differs from the frozen "
            "logging distribution."
        )
    return frontier, selected, stored_propensity


def _validate_selection(
    decision: sqlite3.Row,
    event: sqlite3.Row,
) -> _DecisionBoundary:
    decision_id = decision["id"]
    label = f"QuestionSelected event {event['event_id']}"
    policy_version = _require_nonblank(
        decision["policy_version"],
        f"Decision {decision_id} policy_version",
    )
    if policy_version not in SUPPORTED_LOGGING_POLICY_VERSIONS:
        raise ValidationError(
            f"Decision {decision_id} uses unsupported logging policy "
            f"{policy_version!r}; supported versions are "
            f"{list(SUPPORTED_LOGGING_POLICY_VERSIONS)!r}."
        )
    frontier, selected, propensity = _validate_frontier(decision)
    payload = _strict_object(event["payload_json"], f"{label} payload")
    metadata = _strict_object(event["metadata_json"], f"{label} metadata")
    _require_exact_fields(
        metadata, QUESTION_SELECTED_METADATA_FIELDS, f"{label} metadata"
    )
    model_version = metadata["learner_model_version"]
    if type(model_version) is not str or model_version not in SUPPORTED_MODEL_VERSIONS:
        raise ValidationError(
            f"{label} uses unsupported learner model {model_version!r}."
        )
    objective_aware = bool(
        decision["question_objective_id"] is not None
        or decision["focus_objective_id"] is not None
    )
    try:
        expected_schema = question_selected_schema_for(
            model_version, objective_aware=objective_aware
        )
    except ValueError as exc:
        raise ValidationError(f"{label} has no supported model boundary.") from exc
    if event["schema_version"] != expected_schema:
        raise ValidationError(
            f"{label} schema does not match learner model {model_version}."
        )
    _require_exact_fields(
        payload,
        (
            QUESTION_SELECTED_OBJECTIVE_FIELDS
            if objective_aware
            else QUESTION_SELECTED_BASE_FIELDS
        ),
        f"{label} payload",
    )
    option_order = _strict_array(
        decision["option_order_json"],
        f"Decision {decision_id} option order",
    )
    if (
        any(type(option_id) is not str or not option_id for option_id in option_order)
        or len(option_order) != len(set(option_order))
    ):
        raise ValidationError(
            f"Decision {decision_id} option order must contain unique IDs."
        )
    expected_payload: dict[str, Any] = {
        "decision_id": decision_id,
        "question_id": decision["question_id"],
        "phase": decision["phase"],
        "candidate_count": decision["candidate_count"],
        "candidate_digest": decision["candidate_digest"],
        "propensity": propensity,
        "score": selected.terms(),
        "option_order": option_order,
        "question_version": decision["question_version"],
        "question_content_hash": decision["question_content_hash"],
        "question_status": decision["question_status"],
        "evidence_weight": decision["evidence_weight"],
        "corpus_release_id": decision["corpus_release_id"],
        "session_revision": decision["session_revision"],
        "learner_revision": decision["learner_revision"],
        "focus_concept_id": decision["focus_concept_id"],
        "focus_misconception_id": decision["focus_misconception_id"],
        "pedagogical_role": decision["pedagogical_role"],
        "focus_valid": bool(decision["focus_valid"]),
    }
    if objective_aware:
        expected_payload.update(
            {
                "question_objective_id": decision["question_objective_id"],
                "focus_objective_id": decision["focus_objective_id"],
            }
        )
    if not same_json_value(payload, expected_payload):
        raise ValidationError(f"{label} payload does not match its decision.")
    if (
        metadata["policy_version"] != policy_version
        or metadata["corpus_release_id"] != decision["corpus_release_id"]
        or event["event_type"] != "QuestionSelected"
        or event["learner_id"] != decision["learner_id"]
        or event["session_id"] != decision["session_id"]
        or event["stream_id"] != f"learner:{decision['learner_id']}"
    ):
        raise ValidationError(f"{label} does not match its decision boundary.")
    if expected_schema == 3 and event["occurred_at"] != decision["created_at"]:
        raise ValidationError(
            f"{label} occurrence time does not match its decision."
        )
    return _DecisionBoundary(
        decision_id=decision_id,
        session_id=decision["session_id"],
        learner_id=decision["learner_id"],
        prediction=selected.predicted_correct,
        propensity=propensity,
        frontier_size=len(frontier),
        model_version=model_version,
    )


def _validate_response(
    connection: sqlite3.Connection,
    decision: sqlite3.Row,
    attempt: sqlite3.Row,
    boundary: _DecisionBoundary,
) -> _Observation:
    decision_id = decision["id"]
    label = f"Attempt {attempt['id']}"
    for field in ("session_id", "learner_id", "question_id"):
        if attempt[field] != decision[field]:
            raise ValidationError(f"{label} {field} differs from its decision.")
    if (
        attempt["decision_id"] != decision_id
        or attempt["question_version"] != decision["question_version"]
        or attempt["presented_order_json"] != decision["option_order_json"]
    ):
        raise ValidationError(f"{label} does not match its immutable decision.")
    is_correct = _require_int(
        attempt["is_correct"], f"{label} is_correct"
    )
    if is_correct not in {0, 1}:
        raise ValidationError(f"{label} is_correct must be zero or one.")
    selected_option_id = attempt["selected_option_id"]
    selected_misconception_id: str | None = None
    if selected_option_id is None:
        if is_correct:
            raise ValidationError(f"{label} cannot abstain correctly.")
    else:
        _require_nonblank(selected_option_id, f"{label} selected_option_id")
        option = connection.execute(
            """SELECT is_correct, misconception_id FROM options
               WHERE question_id=? AND option_id=?""",
            (decision["question_id"], selected_option_id),
        ).fetchone()
        if option is None:
            raise ValidationError(f"{label} selects an unknown option.")
        if option["is_correct"] != is_correct:
            raise ValidationError(f"{label} correctness differs from its option.")
        selected_misconception_id = option["misconception_id"]
    confidence = attempt["confidence"]
    if confidence is not None:
        confidence = _require_probability(
            confidence, f"{label} confidence", positive=False
        )
    response_ms = attempt["response_ms"]
    if response_ms is not None:
        response_ms = _require_int(
            response_ms, f"{label} response_ms"
        )
    hint_count = _require_int(attempt["hint_count"], f"{label} hint_count")
    feedback_shown = _require_int(
        attempt["feedback_shown"], f"{label} feedback_shown"
    )
    if feedback_shown not in {0, 1}:
        raise ValidationError(f"{label} feedback_shown must be zero or one.")
    response = connection.execute(
        "SELECT * FROM events WHERE event_id=?",
        (attempt["event_id"],),
    ).fetchone()
    if response is None or response["event_type"] != "ResponseSubmitted":
        raise ValidationError(f"{label} lacks its ResponseSubmitted event.")
    event_label = f"ResponseSubmitted event {response['event_id']}"
    if response["schema_version"] not in {1, 2}:
        raise ValidationError(
            f"{event_label} uses unsupported schema {response['schema_version']}."
        )
    payload = _strict_object(
        response["payload_json"], f"{event_label} payload"
    )
    metadata = _strict_object(
        response["metadata_json"], f"{event_label} metadata"
    )
    _require_exact_fields(payload, RESPONSE_FIELDS, f"{event_label} payload")
    expected_metadata_fields = (
        RESPONSE_METADATA_FIELDS_WITH_MISCONCEPTION_ALGORITHM
        if response["schema_version"] == 2
        else RESPONSE_METADATA_FIELDS
    )
    _require_exact_fields(
        metadata, expected_metadata_fields, f"{event_label} metadata"
    )
    if (
        response["schema_version"] == 2
        and metadata[MISCONCEPTION_ALGORITHM_METADATA_KEY]
        != MISCONCEPTION_ALGORITHM_VERSION
    ):
        raise ValidationError(
            f"{event_label} uses unsupported misconception inference."
        )
    presented_order = _strict_array(
        attempt["presented_order_json"], f"{label} presented order"
    )
    expected_payload = {
        "decision_id": decision_id,
        "question_id": decision["question_id"],
        "question_version": decision["question_version"],
        "selected_option_id": selected_option_id,
        "is_correct": bool(is_correct),
        "confidence": confidence,
        "response_ms": response_ms,
        "hint_count": hint_count,
        "feedback_shown": bool(feedback_shown),
        "presented_order": presented_order,
    }
    if not same_json_value(payload, expected_payload):
        raise ValidationError(
            f"{event_label} payload differs from its attempt."
        )
    expected_metadata: dict[str, Any] = {
        "policy_version": decision["policy_version"],
        "learner_model_version": boundary.model_version,
        "corpus_release_id": decision["corpus_release_id"],
        "question_content_hash": decision["question_content_hash"],
        "question_status": decision["question_status"],
        "evidence_weight": decision["evidence_weight"],
        "selection_learner_revision": decision["learner_revision"],
        "application_learner_revision": decision["learner_revision"],
    }
    if response["schema_version"] == 2:
        expected_metadata[MISCONCEPTION_ALGORITHM_METADATA_KEY] = (
            MISCONCEPTION_ALGORITHM_VERSION
        )
    if not same_json_value(metadata, expected_metadata):
        raise ValidationError(
            f"{event_label} metadata differs from its decision boundary."
        )
    if (
        response["learner_id"] != decision["learner_id"]
        or response["session_id"] != decision["session_id"]
        or response["stream_id"] != f"learner:{decision['learner_id']}"
        or response["causation_id"] != decision_id
        or response["occurred_at"] != attempt["answered_at"]
    ):
        raise ValidationError(
            f"{event_label} envelope differs from its attempt."
        )
    response_class = classify_response_for_model(
        model_version=boundary.model_version,
        correct=bool(is_correct),
        selected_option_id=selected_option_id,
        selected_misconception_id=selected_misconception_id,
        confidence=confidence,
        response_ms=response_ms,
        hint_count=hint_count,
    )
    return _Observation(
        decision_id=decision_id,
        prediction=boundary.prediction,
        correct=is_correct,
        credible_retrieval=int(
            response_class is ResponseClass.CREDIBLE_SUCCESS
        ),
        propensity=boundary.propensity,
        target_probability=1.0 / boundary.frontier_size,
    )


def _calibration(observations: list[_Observation]) -> dict[str, Any]:
    if not observations:
        return {
            "count": 0,
            "brier_score": None,
            "log_loss": None,
            "expected_calibration_error": None,
            "bin_count": CALIBRATION_BIN_COUNT,
            "bins": [],
            "outcome": "raw correctness; abstention is incorrect",
        }
    count = len(observations)
    brier = sum(
        (observation.prediction - observation.correct) ** 2
        for observation in observations
    ) / count
    log_loss = -sum(
        observation.correct
        * math.log(
            min(
                1.0 - LOG_LOSS_CLIP,
                max(LOG_LOSS_CLIP, observation.prediction),
            )
        )
        + (1 - observation.correct)
        * math.log(
            min(
                1.0 - LOG_LOSS_CLIP,
                max(LOG_LOSS_CLIP, 1.0 - observation.prediction),
            )
        )
        for observation in observations
    ) / count
    buckets: list[list[_Observation]] = [
        [] for _ in range(CALIBRATION_BIN_COUNT)
    ]
    for observation in observations:
        index = min(
            CALIBRATION_BIN_COUNT - 1,
            int(observation.prediction * CALIBRATION_BIN_COUNT),
        )
        buckets[index].append(observation)
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        predicted = sum(item.prediction for item in bucket) / len(bucket)
        observed = sum(item.correct for item in bucket) / len(bucket)
        ece += len(bucket) / count * abs(predicted - observed)
        bins.append(
            {
                "lower": index / CALIBRATION_BIN_COUNT,
                "upper": (index + 1) / CALIBRATION_BIN_COUNT,
                "count": len(bucket),
                "mean_predicted": predicted,
                "observed_accuracy": observed,
            }
        )
    return {
        "count": count,
        "brier_score": brier,
        "log_loss": log_loss,
        "expected_calibration_error": ece,
        "bin_count": CALIBRATION_BIN_COUNT,
        "bins": bins,
        "outcome": "raw correctness; abstention is incorrect",
    }


def _weighted_reward(
    observations: list[_Observation],
    reward_field: str,
) -> dict[str, Any]:
    if not observations:
        return {
            "behavior_mean": None,
            "ips": None,
            "snips": None,
        }
    weights = [observation.importance_weight for observation in observations]
    rewards = [
        int(getattr(observation, reward_field))
        for observation in observations
    ]
    weighted_sum = sum(
        weight * reward
        for weight, reward in zip(weights, rewards, strict=True)
    )
    return {
        "behavior_mean": sum(rewards) / len(rewards),
        "ips": weighted_sum / len(rewards),
        "snips": weighted_sum / sum(weights),
    }


def _uniform_safe_frontier(
    observations: list[_Observation],
) -> dict[str, Any]:
    if not observations:
        return {
            "policy_version": UNIFORM_SAFE_FRONTIER_POLICY_VERSION,
            "description": (
                "Uniform random choice over the same logged safe frontier."
            ),
            "frontier_size_rule": "K = min(5, candidate_count)",
            "observation_count": 0,
            "status": "unavailable",
            "low_information": True,
            "low_information_rule": "ESS < 30 or ESS/N < 0.10",
            "information_scope": (
                "Importance-weight concentration only; does not assess "
                "independent sample size, response censoring, serial "
                "dependence, or transport to humans."
            ),
            "low_information_reasons": ["no answered logged decisions"],
            "weights": {
                "count": 0,
                "sum": 0.0,
                "mean": None,
                "effective_sample_size": 0.0,
                "effective_sample_ratio": None,
                "effective_sample_size_kind": (
                    "Kish importance-weight concentration diagnostic"
                ),
                "dependence_adjusted": False,
                "maximum": None,
                "p95": None,
                "zero_count": 0,
                "support_violations": 0,
            },
            "raw_correctness": _weighted_reward([], "correct"),
            "credible_retrieval": _weighted_reward(
                [], "credible_retrieval"
            ),
        }
    weights = [observation.importance_weight for observation in observations]
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValidationError("Importance weights must be finite and non-negative.")
    count = len(weights)
    weight_sum = sum(weights)
    squared_sum = sum(weight * weight for weight in weights)
    if weight_sum <= 0.0 or squared_sum <= 0.0:
        raise ValidationError(
            "Logged decisions provide no supported target-policy mass."
        )
    effective_sample_size = weight_sum * weight_sum / squared_sum
    effective_sample_ratio = effective_sample_size / count
    ordered = sorted(weights)
    p95 = ordered[math.ceil(0.95 * count) - 1]
    reasons: list[str] = []
    if effective_sample_size < MIN_EFFECTIVE_SAMPLE_SIZE:
        reasons.append("effective sample size is below 30")
    if effective_sample_ratio < MIN_EFFECTIVE_SAMPLE_RATIO:
        reasons.append("effective sample ratio is below 0.10")
    low_information = bool(reasons)
    return {
        "policy_version": UNIFORM_SAFE_FRONTIER_POLICY_VERSION,
        "description": (
            "Uniform random choice over the same logged safe frontier."
        ),
        "frontier_size_rule": "K = min(5, candidate_count)",
        "observation_count": count,
        "status": "low_information" if low_information else "descriptive_only",
        "low_information": low_information,
        "low_information_rule": "ESS < 30 or ESS/N < 0.10",
        "information_scope": (
            "Importance-weight concentration only; does not assess "
            "independent sample size, response censoring, serial dependence, "
            "or transport to humans."
        ),
        "low_information_reasons": reasons,
        "weights": {
            "count": count,
            "sum": weight_sum,
            "mean": weight_sum / count,
            "effective_sample_size": effective_sample_size,
            "effective_sample_ratio": effective_sample_ratio,
            "effective_sample_size_kind": (
                "Kish importance-weight concentration diagnostic"
            ),
            "dependence_adjusted": False,
            "maximum": max(weights),
            "p95": p95,
            "zero_count": sum(weight == 0.0 for weight in weights),
            "support_violations": 0,
        },
        "raw_correctness": _weighted_reward(observations, "correct"),
        "credible_retrieval": _weighted_reward(
            observations, "credible_retrieval"
        ),
    }


def _outcome_summary(
    *,
    raw_correct: int,
    credible: int,
    observed: int,
) -> dict[str, Any]:
    return {
        "observed_same_action_count": observed,
        "raw_correct_count": raw_correct,
        "raw_correct_rate": (
            raw_correct / observed if observed else None
        ),
        "credible_retrieval_count": credible,
        "credible_retrieval_rate": (
            credible / observed if observed else None
        ),
    }


def _prospective_shadow(
    connection: sqlite3.Connection,
    *,
    decision_ids: set[str],
    observations_by_decision: Mapping[str, _Observation],
) -> dict[str, Any]:
    errors = policy_shadow_integrity_errors(connection)
    if errors:
        raise ValidationError(
            "Policy-shadow event/projection integrity failed: " + "; ".join(errors)
        )
    snapshot = policy_shadow_projection_snapshot(connection)
    rows = [
        row
        for row in snapshot["evaluations"]
        if row["decision_id"] in decision_ids
    ]
    seen: set[tuple[str, str]] = set()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["decision_id"], row["challenger_policy_version"])
        if key in seen:
            raise ValidationError(
                "A decision has duplicate prospective evaluations for "
                f"challenger {row['challenger_policy_version']!r}."
            )
        seen.add(key)
        if row["logging_policy_version"] not in SUPPORTED_LOGGING_POLICY_VERSIONS:
            raise ValidationError(
                "Policy-shadow evaluation uses an unsupported logging policy "
                f"{row['logging_policy_version']!r}."
            )
        if (
            row["shadow_only"] != 1
            or row["selection_applied"] != 0
            or row["mastery_applied"] != 0
        ):
            raise ValidationError(
                f"Policy-shadow evaluation {row['id']} violates isolation."
            )
        grouped[row["challenger_policy_version"]].append(row)

    def summarize(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
        agreements = [row for row in group_rows if row["agreement"] == 1]
        divergences = [row for row in group_rows if row["agreement"] == 0]
        same_observations = [
            observations_by_decision[row["decision_id"]]
            for row in agreements
            if row["decision_id"] in observations_by_decision
        ]
        divergent_observed = sum(
            row["decision_id"] in observations_by_decision
            for row in divergences
        )
        evaluation_count = len(group_rows)
        return {
            "evaluation_count": evaluation_count,
            "agreement_count": len(agreements),
            "agreement_rate": (
                len(agreements) / evaluation_count
                if evaluation_count
                else None
            ),
            "divergence_count": len(divergences),
            "divergence_rate": (
                len(divergences) / evaluation_count
                if evaluation_count
                else None
            ),
            "unanswered_same_action_count": (
                len(agreements) - len(same_observations)
            ),
            "divergent_observed_outcomes_withheld": divergent_observed,
            "same_action_outcomes": _outcome_summary(
                raw_correct=sum(item.correct for item in same_observations),
                credible=sum(
                    item.credible_retrieval for item in same_observations
                ),
                observed=len(same_observations),
            ),
        }

    challenger_summaries = [
        {
            "challenger_policy_version": challenger,
            **summarize(grouped[challenger]),
        }
        for challenger in sorted(grouped)
    ]
    overall = summarize(rows)
    return {
        "shadow_contract_version": POLICY_SHADOW_CONTRACT_VERSION,
        **overall,
        "challengers": challenger_summaries,
        "outcome_rule": (
            "Outcomes are reported only for answered decisions when "
            "challenger_question_id equals the actually served "
            "live_question_id."
        ),
        "response_censoring_adjusted": False,
    }


def _scope_clause(
    *,
    session_id: str | None,
    learner_id: str | None,
    alias: str,
) -> tuple[str, tuple[str, ...]]:
    predicates: list[str] = []
    parameters: list[str] = []
    if session_id is not None:
        predicates.append(f"{alias}.session_id=?")
        parameters.append(session_id)
    if learner_id is not None:
        predicates.append(f"{alias}.learner_id=?")
        parameters.append(learner_id)
    return (
        (" WHERE " + " AND ".join(predicates)) if predicates else "",
        tuple(parameters),
    )


def _validate_scope(
    connection: sqlite3.Connection,
    *,
    session_id: str | None,
    learner_id: str | None,
) -> None:
    if session_id is not None:
        _require_nonblank(session_id, "session_id")
        session = connection.execute(
            "SELECT learner_id FROM sessions WHERE id=?", (session_id,)
        ).fetchone()
        if session is None:
            raise NotFoundError(f"Unknown session: {session_id}")
        if learner_id is not None and session["learner_id"] != learner_id:
            raise ValidationError(
                "The requested session does not belong to the requested learner."
            )
    if learner_id is not None:
        _require_nonblank(learner_id, "learner_id")
        if (
            connection.execute(
                "SELECT 1 FROM learners WHERE id=?", (learner_id,)
            ).fetchone()
            is None
        ):
            raise NotFoundError(f"Unknown learner: {learner_id}")


def build_policy_shadow_report(
    database: Database,
    *,
    session_id: str | None = None,
    learner_id: str | None = None,
) -> dict[str, Any]:
    """Build a strictly validated, observational policy report.

    The database is opened through SQLite's read-only mode.  Unknown logging
    versions, malformed event fields, broken event/projection pairs, and
    invalid propensities raise :class:`ValidationError` rather than producing a
    partial estimate.
    """

    if not isinstance(database, Database):
        raise ValidationError("database must be a TSQ Database.")
    if session_id is not None:
        _require_nonblank(session_id, "session_id")
    if learner_id is not None:
        _require_nonblank(learner_id, "learner_id")
    if database.path == Path(":memory:"):
        reader = database
    else:
        reader = Database(database.path, read_only=True)
    try:
        with reader.read() as connection:
            # Pin one read snapshot so a concurrent answer or selection cannot
            # make the event, decision, and projection queries disagree.
            connection.execute("BEGIN")
            _validate_scope(
                connection,
                session_id=session_id,
                learner_id=learner_id,
            )
            decision_clause, decision_parameters = _scope_clause(
                session_id=session_id,
                learner_id=learner_id,
                alias="decision",
            )
            decisions = connection.execute(
                "SELECT decision.* FROM decisions decision"
                + decision_clause
                + " ORDER BY decision.created_at, decision.id",
                decision_parameters,
            ).fetchall()
            decision_by_id = {row["id"]: row for row in decisions}

            event_clause, event_parameters = _scope_clause(
                session_id=session_id,
                learner_id=learner_id,
                alias="event",
            )
            selection_query = (
                "SELECT event.* FROM events event "
                "WHERE event.event_type='QuestionSelected'"
            )
            if event_clause:
                selection_query += " AND " + event_clause.removeprefix(" WHERE ")
            selection_query += " ORDER BY event.stream_id, event.stream_version"
            selection_events = connection.execute(
                selection_query, event_parameters
            ).fetchall()
            selections_by_decision: dict[str, list[sqlite3.Row]] = defaultdict(
                list
            )
            for event in selection_events:
                payload = _strict_object(
                    event["payload_json"],
                    f"QuestionSelected event {event['event_id']} payload",
                )
                decision_id = _require_nonblank(
                    payload.get("decision_id"),
                    f"QuestionSelected event {event['event_id']} decision_id",
                )
                if decision_id not in decision_by_id:
                    raise ValidationError(
                        f"QuestionSelected event {event['event_id']} has no "
                        "decision in the requested scope."
                    )
                selections_by_decision[decision_id].append(event)

            attempt_rows = connection.execute(
                """SELECT attempt.* FROM attempts attempt
                   JOIN decisions decision ON decision.id=attempt.decision_id"""
                + decision_clause
                + " ORDER BY attempt.answered_at, attempt.id",
                decision_parameters,
            ).fetchall()
            attempts_by_decision: dict[str, sqlite3.Row] = {}
            for attempt in attempt_rows:
                if attempt["decision_id"] in attempts_by_decision:
                    raise ValidationError(
                        f"Decision {attempt['decision_id']} has multiple attempts."
                    )
                attempts_by_decision[attempt["decision_id"]] = attempt

            observations: list[_Observation] = []
            invalidated_count = 0
            pending_count = 0
            for decision in decisions:
                decision_id = decision["id"]
                selection_matches = selections_by_decision.get(decision_id, [])
                if len(selection_matches) != 1:
                    raise ValidationError(
                        f"Decision {decision_id} requires exactly one "
                        f"QuestionSelected event; found {len(selection_matches)}."
                    )
                boundary = _validate_selection(
                    decision, selection_matches[0]
                )
                attempt = attempts_by_decision.get(decision_id)
                invalidated = decision["invalidated_at"] is not None
                if invalidated:
                    invalidated_count += 1
                    if attempt is not None:
                        raise ValidationError(
                            f"Invalidated decision {decision_id} has an attempt."
                        )
                if attempt is None:
                    if decision["consumed_at"] is not None:
                        raise ValidationError(
                            f"Decision {decision_id} is consumed without an attempt."
                        )
                    if not invalidated:
                        pending_count += 1
                    continue
                if decision["consumed_at"] is None:
                    raise ValidationError(
                        f"Decision {decision_id} has an unconsumed attempt."
                    )
                observations.append(
                    _validate_response(
                        connection, decision, attempt, boundary
                    )
                )

            observations_by_decision = {
                observation.decision_id: observation
                for observation in observations
            }
            prospective = _prospective_shadow(
                connection,
                decision_ids=set(decision_by_id),
                observations_by_decision=observations_by_decision,
            )
    except sqlite3.DatabaseError as exc:
        raise ValidationError(
            f"Policy-shadow reporting could not read the database: {exc}"
        ) from exc

    return {
        "report_version": POLICY_SHADOW_REPORT_VERSION,
        "scope": {
            "session_id": session_id,
            "learner_id": learner_id,
        },
        "supported_logging_policy_versions": list(
            SUPPORTED_LOGGING_POLICY_VERSIONS
        ),
        "decision_counts": {
            "total": len(decisions),
            "answered": len(observations),
            "pending": pending_count,
            "invalidated": invalidated_count,
        },
        "behavior_calibration": _calibration(observations),
        "uniform_safe_frontier": _uniform_safe_frontier(observations),
        "prospective_shadow": prospective,
        "inference_boundary": {
            "shadow_only": True,
            "affects_selection": False,
            "affects_mastery": False,
            "estimand": (
                "one-step complete-case operational response outcomes among "
                "answered behavior-policy-visited decision states"
            ),
            "answered_complete_cases_only": True,
            "response_censoring_adjusted": False,
            "sequential_dependence_adjusted": False,
            "importance_weight_ess_is_independent_sample_size": False,
            "counterfactual_trajectory": False,
            "causal_learning_effect": False,
            "retention_inference": False,
            "productive_skill_inference": False,
            "posterior_delta_used_as_reward": False,
            "historical_frontier_provenance": {
                "v17": (
                    "Top-candidate prefixes are mutable relational audit rows "
                    "stored beside an immutable candidate digest that cannot "
                    "be recomputed from the retained prefix alone."
                ),
                "v18": (
                    "Safe frontiers used by prospective shadow evaluation are "
                    "event-backed and projection-checked."
                ),
                "shared_limit": (
                    "Neither boundary reconstructs full unlogged candidate "
                    "sets or counterfactual trajectories."
                ),
            },
            "not_inferred": [
                "answers to unchosen questions",
                "outcomes for unanswered selected questions",
                "selection-dependent response or abandonment effects",
                "independent evidence size under repeated learner, session, "
                "family, or temporal dependence",
                "counterfactual remediation or learner-state trajectories",
                "causal teaching benefit",
                "delayed transfer or retention",
                "implementation, explanation, design, or project skill",
            ],
        },
    }


# A concise alias is useful to analysis callers while the explicit ``build_*``
# name remains the public integration boundary for the CLI.
policy_shadow_report = build_policy_shadow_report


__all__ = [
    "CALIBRATION_BIN_COUNT",
    "MIN_EFFECTIVE_SAMPLE_RATIO",
    "MIN_EFFECTIVE_SAMPLE_SIZE",
    "POLICY_SHADOW_REPORT_VERSION",
    "SUPPORTED_LOGGING_POLICY_VERSIONS",
    "UNIFORM_SAFE_FRONTIER_POLICY_VERSION",
    "build_policy_shadow_report",
    "policy_shadow_report",
]
