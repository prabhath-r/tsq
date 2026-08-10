# SPDX-License-Identifier: MPL-2.0

"""Frozen learner-model capabilities and projection format identities.

``MODEL_VERSION`` is intentionally not defined here.  Callers may expose a
default-model alias, but immutable history must always dispatch through one of
the explicit version constants or capability maps below.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


LEGACY_MODEL_VERSION = "irt-gaussian-retention-v3"
CONCEPT_MODEL_VERSION = "irt-gaussian-retention-v4"
OBJECTIVE_GAUSSIAN_MODEL_VERSION = "irt-gaussian-retention-v5"
OBJECTIVE_GRID_V6_MODEL_VERSION = "irt-grid-survival-v6"
OBJECTIVE_GRID_V7_MODEL_VERSION = "irt-grid-spacing-v7"
OBJECTIVE_GRID_V8_MODEL_VERSION = "irt-grid-window-v8"
CANONICAL_FAMILY_V9_MODEL_VERSION = "irt-grid-family-equivalence-v9"
DEFAULT_LEARNER_MODEL_VERSION = CANONICAL_FAMILY_V9_MODEL_VERSION

LEGACY_QUESTION_SELECTED_EVENT_SCHEMA_VERSIONS = frozenset({1, 2})
BOUND_QUESTION_SELECTED_EVENT_SCHEMA_VERSION = 3
SUPPORTED_QUESTION_SELECTED_EVENT_SCHEMA_VERSIONS = frozenset(
    {
        *LEGACY_QUESTION_SELECTED_EVENT_SCHEMA_VERSIONS,
        BOUND_QUESTION_SELECTED_EVENT_SCHEMA_VERSION,
    }
)

PRE_GRID_MODEL_VERSIONS = frozenset(
    {
        LEGACY_MODEL_VERSION,
        CONCEPT_MODEL_VERSION,
        OBJECTIVE_GAUSSIAN_MODEL_VERSION,
    }
)
OBJECTIVE_GRID_MODEL_VERSIONS = frozenset(
    {
        OBJECTIVE_GRID_V6_MODEL_VERSION,
        OBJECTIVE_GRID_V7_MODEL_VERSION,
        OBJECTIVE_GRID_V8_MODEL_VERSION,
        CANONICAL_FAMILY_V9_MODEL_VERSION,
    }
)
SPACING_AWARE_FAMILY_MODEL_VERSIONS = frozenset(
    {
        OBJECTIVE_GRID_V7_MODEL_VERSION,
        OBJECTIVE_GRID_V8_MODEL_VERSION,
        CANONICAL_FAMILY_V9_MODEL_VERSION,
    }
)
CANONICAL_FAMILY_MODEL_VERSIONS = frozenset(
    {CANONICAL_FAMILY_V9_MODEL_VERSION}
)
AUTHORITATIVE_RESPONSE_WINDOW_MODEL_VERSIONS = frozenset(
    {OBJECTIVE_GRID_V8_MODEL_VERSION, CANONICAL_FAMILY_V9_MODEL_VERSION}
)
COMPLETE_TRANSITION_OUTCOME_MODEL_VERSIONS = frozenset(
    {OBJECTIVE_GRID_V8_MODEL_VERSION, CANONICAL_FAMILY_V9_MODEL_VERSION}
)
OBJECTIVE_MODEL_VERSIONS = frozenset(
    {OBJECTIVE_GAUSSIAN_MODEL_VERSION, *OBJECTIVE_GRID_MODEL_VERSIONS}
)
SUPPORTED_MODEL_VERSIONS = frozenset(
    {
        LEGACY_MODEL_VERSION,
        CONCEPT_MODEL_VERSION,
        *OBJECTIVE_MODEL_VERSIONS,
    }
)


@dataclass(frozen=True, slots=True)
class ResponseTelemetryContract:
    """Observable fields required before one response is behaviorally credible."""

    confidence_required: bool
    response_time_required: bool
    named_error_confidence_required: bool


_OPTIONAL_TELEMETRY = ResponseTelemetryContract(
    confidence_required=False,
    response_time_required=False,
    named_error_confidence_required=False,
)
_V6_TELEMETRY = ResponseTelemetryContract(
    confidence_required=False,
    response_time_required=True,
    named_error_confidence_required=False,
)
_V7_TELEMETRY = ResponseTelemetryContract(
    confidence_required=True,
    response_time_required=True,
    named_error_confidence_required=True,
)
RESPONSE_TELEMETRY_CONTRACTS: Mapping[str, ResponseTelemetryContract] = (
    MappingProxyType(
        {
            LEGACY_MODEL_VERSION: _OPTIONAL_TELEMETRY,
            CONCEPT_MODEL_VERSION: _OPTIONAL_TELEMETRY,
            OBJECTIVE_GAUSSIAN_MODEL_VERSION: _OPTIONAL_TELEMETRY,
            OBJECTIVE_GRID_V6_MODEL_VERSION: _V6_TELEMETRY,
            OBJECTIVE_GRID_V7_MODEL_VERSION: _V7_TELEMETRY,
            OBJECTIVE_GRID_V8_MODEL_VERSION: _V7_TELEMETRY,
            CANONICAL_FAMILY_V9_MODEL_VERSION: _V7_TELEMETRY,
        }
    )
)


@dataclass(frozen=True, slots=True)
class ProjectionFormat:
    event_schema_version: int
    hash_version: int


CONCEPT_PROJECTION_FORMAT = ProjectionFormat(
    event_schema_version=2,
    hash_version=1,
)
OBJECTIVE_PROJECTION_FORMATS: Mapping[str, ProjectionFormat] = MappingProxyType(
    {
        OBJECTIVE_GAUSSIAN_MODEL_VERSION: ProjectionFormat(
            event_schema_version=3,
            hash_version=2,
        ),
        OBJECTIVE_GRID_V6_MODEL_VERSION: ProjectionFormat(
            event_schema_version=4,
            hash_version=3,
        ),
        OBJECTIVE_GRID_V7_MODEL_VERSION: ProjectionFormat(
            event_schema_version=4,
            hash_version=3,
        ),
        OBJECTIVE_GRID_V8_MODEL_VERSION: ProjectionFormat(
            event_schema_version=4,
            hash_version=3,
        ),
        CANONICAL_FAMILY_V9_MODEL_VERSION: ProjectionFormat(
            event_schema_version=4,
            hash_version=3,
        ),
    }
)
PROJECTION_HASH_VERSION_BY_EVENT_SCHEMA: Mapping[int, int] = MappingProxyType(
    {
        1: 1,
        2: 1,
        3: 2,
        4: 3,
    }
)
PROJECTION_MODEL_VERSIONS_BY_EVENT_SCHEMA: Mapping[int, frozenset[str]] = (
    MappingProxyType(
        {
            3: frozenset({OBJECTIVE_GAUSSIAN_MODEL_VERSION}),
            4: OBJECTIVE_GRID_MODEL_VERSIONS,
        }
    )
)


def projection_format_for(
    model_version: str, *, objective_aware: bool
) -> ProjectionFormat:
    """Resolve a writer format without interpreting a mutable default alias."""

    if model_version not in SUPPORTED_MODEL_VERSIONS:
        raise ValueError(f"Unsupported learner model version: {model_version}")
    if not objective_aware:
        return CONCEPT_PROJECTION_FORMAT
    try:
        return OBJECTIVE_PROJECTION_FORMATS[model_version]
    except KeyError as exc:
        raise ValueError(
            f"Learner model {model_version} has no objective projection format."
        ) from exc


def question_selected_schema_for(
    model_version: str, *, objective_aware: bool
) -> int:
    """Return the immutable selection-event envelope for one model.

    Schema 3 binds the event occurrence time to the persisted decision clock.
    Earlier schemas remain readable exactly as written, but they must never be
    retroactively treated as having that authoritative timing guarantee.
    """

    if model_version not in SUPPORTED_MODEL_VERSIONS:
        raise ValueError(f"Unsupported learner model version: {model_version}")
    if objective_aware and model_version not in OBJECTIVE_MODEL_VERSIONS:
        raise ValueError(
            f"Learner model {model_version} cannot select objective-aware questions."
        )
    if model_version in AUTHORITATIVE_RESPONSE_WINDOW_MODEL_VERSIONS:
        return BOUND_QUESTION_SELECTED_EVENT_SCHEMA_VERSION
    return 2 if objective_aware else 1
