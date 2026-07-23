# SPDX-License-Identifier: MPL-2.0

"""Closed field contracts for immutable adaptive-learning events.

The event envelope carries a schema version, so accepting undeclared payload
or metadata fields would silently change historical semantics without a schema
change.  Integrity checking and replay import these same frozen sets.
"""

from __future__ import annotations

from math import isfinite
from typing import Any


def same_json_value(actual: Any, expected: Any) -> bool:
    """Compare JSON values without Python's bool/int equality collision.

    SQLite may faithfully project a JSON numeric as either ``int`` or
    ``float``, so finite non-boolean numbers compare by value. Every other JSON
    type—including nested objects and arrays—must retain its exact kind.
    """

    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is bool and type(expected) is bool and actual == expected
    if (
        isinstance(actual, (int, float))
        and isinstance(expected, (int, float))
    ):
        return (
            (not isinstance(actual, float) or isfinite(actual))
            and (not isinstance(expected, float) or isfinite(expected))
            and actual == expected
        )
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return set(actual) == set(expected) and all(
            same_json_value(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(
            same_json_value(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


QUESTION_SELECTED_METADATA_FIELDS = frozenset(
    {"policy_version", "learner_model_version", "corpus_release_id"}
)
QUESTION_SELECTED_BASE_FIELDS = frozenset(
    {
        "decision_id",
        "question_id",
        "phase",
        "candidate_count",
        "candidate_digest",
        "propensity",
        "score",
        "option_order",
        "question_version",
        "question_content_hash",
        "question_status",
        "evidence_weight",
        "corpus_release_id",
        "session_revision",
        "learner_revision",
        "focus_concept_id",
        "focus_misconception_id",
        "pedagogical_role",
        "focus_valid",
    }
)
QUESTION_SELECTED_OBJECTIVE_FIELDS = QUESTION_SELECTED_BASE_FIELDS | frozenset(
    {"question_objective_id", "focus_objective_id"}
)

ATTEMPT_OUTCOME_BASE_FIELDS = frozenset(
    {
        "interaction_id",
        "correct",
        "selected_option",
        "correct_option",
        "next_phase",
        "focus_concept_id",
        "focus_misconception_id",
        "state_changes",
        "transition_reason",
        "boundary_decision",
    }
)
ATTEMPT_OUTCOME_COMPLETE_FIELDS = ATTEMPT_OUTCOME_BASE_FIELDS | frozenset(
    {"remediation_depth", "remediation_path"}
)
ATTEMPT_OUTCOME_OPTIONAL_FIELDS = frozenset({"focus_objective_id"})

RESPONSE_FIELDS = frozenset(
    {
        "decision_id",
        "question_id",
        "question_version",
        "selected_option_id",
        "is_correct",
        "confidence",
        "response_ms",
        "hint_count",
        "feedback_shown",
        "presented_order",
    }
)
RESPONSE_METADATA_FIELDS = frozenset(
    {
        "policy_version",
        "learner_model_version",
        "corpus_release_id",
        "question_content_hash",
        "question_status",
        "evidence_weight",
        "selection_learner_revision",
        "application_learner_revision",
    }
)
RESPONSE_METADATA_FIELDS_WITH_MISCONCEPTION_ALGORITHM = (
    RESPONSE_METADATA_FIELDS | frozenset({"misconception_algorithm"})
)

PROJECTION_FIELDS_V1 = frozenset(
    {
        "response_event_id",
        "state_changes",
        "phase",
        "focus_concept_id",
        "focus_misconception_id",
        "remediation_depth",
        "remediation_path",
        "corpus_release_id",
        "learner_revision",
        "projection_hash",
    }
)
PROJECTION_FIELDS_V2 = PROJECTION_FIELDS_V1 | frozenset(
    {"transition_reason", "boundary_decision"}
)
PROJECTION_FIELDS_V3 = PROJECTION_FIELDS_V2 | frozenset(
    {
        "question_objective_id",
        "focus_objective_id",
        "projection_hash_version",
    }
)
# Schema four changes the hash semantics, not the compact payload shape.
PROJECTION_FIELDS_V4 = PROJECTION_FIELDS_V3
PROJECTION_FIELDS_BY_SCHEMA = {
    1: PROJECTION_FIELDS_V1,
    2: PROJECTION_FIELDS_V2,
    3: PROJECTION_FIELDS_V3,
    4: PROJECTION_FIELDS_V4,
}
PROJECTION_METADATA_FIELDS = frozenset(
    {"learner_model_version", "corpus_release_id", "evidence_weight"}
)
PROJECTION_METADATA_FIELDS_WITH_MISCONCEPTION_ALGORITHM = (
    PROJECTION_METADATA_FIELDS | frozenset({"misconception_algorithm"})
)

REMEDIATION_TRANSITION_FIELDS_V1 = frozenset(
    {
        "from_phase",
        "to_phase",
        "focus_concept_id",
        "focus_misconception_id",
        "remediation_depth",
        "remediation_path",
        "pedagogical_role",
        "focus_valid",
        "unguided",
    }
)
REMEDIATION_TRANSITION_FIELDS = REMEDIATION_TRANSITION_FIELDS_V1 | frozenset(
    {"transition_reason", "boundary_decision"}
)
REMEDIATION_TRANSITION_OBJECTIVE_FIELDS = (
    REMEDIATION_TRANSITION_FIELDS | frozenset({"focus_objective_id"})
)
REMEDIATION_TRANSITION_METADATA_FIELDS = frozenset(
    {"policy_version", "corpus_release_id"}
)
