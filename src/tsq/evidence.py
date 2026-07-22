# SPDX-License-Identifier: MPL-2.0

"""Pure, deterministic evidence primitives for multimodal learning tasks.

The module deliberately separates three things that are easy to conflate:

* task telemetry records what happened;
* rubric evaluations describe observable performance; and
* evidence records state which reviewed competency claims the observation may
  update.

Telemetry is restricted to meaningful, typed checkpoints and content digests.
Raw code, prose, commands, stdout, and arbitrary callbacks are not accepted.
Nothing in this module executes learner artifacts.  A future runner must remain
an isolated subsystem and pass only validated results into these primitives.

The reducer is intentionally conservative.  Missing telemetry is neutral,
post-feedback work has zero competence-evidence weight, tool use has no direct
positive or negative weight, and dependent observations share finite budgets.
All outputs are deterministic functions of immutable inputs and serialize to
finite canonical JSON for hashing, replay, and later database integration.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
MAX_ACTION_COUNTER = 1_000_000_000

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class TaskModality(StrEnum):
    """The principal way a learner demonstrates a competency."""

    SELECTION = "selection"
    CALCULATION = "calculation"
    EXPLANATION = "explanation"
    DEBUGGING = "debugging"
    IMPLEMENTATION = "implementation"
    DESIGN = "design"
    EXPERIMENT = "experiment"
    CRITIQUE = "critique"
    TRANSFER = "transfer"
    PROJECT = "project"


class ActionKind(StrEnum):
    """Allowlisted semantic checkpoints; none contains learner-authored text."""

    STARTED = "started"
    HINT_REQUESTED = "hint_requested"
    ANSWER_REVISED = "answer_revised"
    ARTIFACT_CHECKPOINT = "artifact_checkpoint"
    EXPLANATION_CHECKPOINT = "explanation_checkpoint"
    CHECK_RUN = "check_run"
    TOOL_USED = "tool_used"
    SUBMITTED = "submitted"
    FEEDBACK_SHOWN = "feedback_shown"
    ABANDONED = "abandoned"


class ActionPhase(StrEnum):
    """Assistance phase in which an action or evaluation occurred."""

    UNASSISTED = "unassisted"
    ASSISTED = "assisted"
    POST_FEEDBACK = "post_feedback"


ACTION_PAYLOAD_CONTRACTS: Mapping[ActionKind, Mapping[str, str]] = MappingProxyType(
    {
        ActionKind.STARTED: MappingProxyType({}),
        ActionKind.HINT_REQUESTED: MappingProxyType(
            {"hint_id": "id", "level": "positive_integer"}
        ),
        ActionKind.ANSWER_REVISED: MappingProxyType(
            {"answer_digest": "sha256"}
        ),
        ActionKind.ARTIFACT_CHECKPOINT: MappingProxyType(
            {"artifact_digest": "sha256", "artifact_kind": "id"}
        ),
        ActionKind.EXPLANATION_CHECKPOINT: MappingProxyType(
            {"explanation_digest": "sha256"}
        ),
        ActionKind.CHECK_RUN: MappingProxyType(
            {
                "check_set_id": "id",
                "passed": "nonnegative_integer",
                "failed": "nonnegative_integer",
                "errored": "nonnegative_integer",
                "skipped": "nonnegative_integer",
                "result_digest": "sha256",
            }
        ),
        ActionKind.TOOL_USED: MappingProxyType(
            {"tool_id": "id", "purpose_code": "id"}
        ),
        ActionKind.SUBMITTED: MappingProxyType(
            {"submission_digest": "sha256"}
        ),
        ActionKind.FEEDBACK_SHOWN: MappingProxyType(
            {"feedback_digest": "sha256"}
        ),
        ActionKind.ABANDONED: MappingProxyType({"reason_code": "id"}),
    }
)


class CriterionScale(StrEnum):
    """Interpretation of a rubric score before normalization to ``[0, 1]``."""

    BINARY = "binary"
    ORDERED = "ordered"
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"


class EvaluationStatus(StrEnum):
    """Whether a scorer produced an interpretable rubric observation."""

    VALID = "valid"
    INVALID = "invalid"
    MISSING = "missing"


class ScorerKind(StrEnum):
    """Provenance class for a criterion evaluation."""

    DETERMINISTIC = "deterministic"
    HUMAN = "human"
    MODEL = "model"
    IMPORTED = "imported"


@dataclass(frozen=True, slots=True)
class ScorerContract:
    """Release-pinnable trust boundary for a deterministic or human scorer.

    A matching ID is not enough: the contract names the exact rubric criteria
    the scorer may certify, and evidence must be linked to allowlisted semantic
    actions (optionally constrained to named check sets or artifact kinds), or
    the contract must require an externally verified attestation digest.  The
    reducer validates structure only; signature and identity verification
    belong at the persistence boundary.
    """

    kind: ScorerKind
    scorer_id: str
    scorer_version: str
    criterion_ids: tuple[str, ...]
    evidence_action_kinds: tuple[ActionKind, ...] = ()
    check_set_ids: tuple[str, ...] | None = None
    artifact_kinds: tuple[str, ...] | None = None
    requires_attestation: bool = False

    def __post_init__(self) -> None:
        _require_enum(self.kind, ScorerKind, "ScorerContract.kind")
        if self.kind not in {ScorerKind.DETERMINISTIC, ScorerKind.HUMAN}:
            raise ValueError(
                "ScorerContract may trust only deterministic or human scorers."
            )
        _require_id(self.scorer_id, "ScorerContract.scorer_id")
        _require_id(self.scorer_version, "ScorerContract.scorer_version")
        criterion_ids = _sorted_unique_ids(
            self.criterion_ids, "ScorerContract.criterion_ids"
        )
        if not criterion_ids:
            raise ValueError("ScorerContract.criterion_ids must not be empty.")
        object.__setattr__(self, "criterion_ids", criterion_ids)
        if not isinstance(self.evidence_action_kinds, tuple) or any(
            not isinstance(kind, ActionKind) for kind in self.evidence_action_kinds
        ):
            raise ValueError(
                "ScorerContract.evidence_action_kinds must contain ActionKind values."
            )
        if len(self.evidence_action_kinds) != len(set(self.evidence_action_kinds)):
            raise ValueError("ScorerContract evidence action kinds must be unique.")
        object.__setattr__(
            self,
            "evidence_action_kinds",
            tuple(sorted(self.evidence_action_kinds, key=lambda kind: kind.value)),
        )
        for name in ("check_set_ids", "artifact_kinds"):
            values = getattr(self, name)
            if values is None:
                continue
            normalized = _sorted_unique_ids(values, f"ScorerContract.{name}")
            if not normalized:
                raise ValueError(f"ScorerContract.{name} must not be empty.")
            object.__setattr__(self, name, normalized)
        if (
            self.check_set_ids is not None
            and ActionKind.CHECK_RUN not in self.evidence_action_kinds
        ):
            raise ValueError(
                "ScorerContract.check_set_ids requires check_run evidence actions."
            )
        if (
            self.artifact_kinds is not None
            and ActionKind.ARTIFACT_CHECKPOINT not in self.evidence_action_kinds
        ):
            raise ValueError(
                "ScorerContract.artifact_kinds requires artifact_checkpoint "
                "evidence actions."
            )
        _require_exact_type(
            self.requires_attestation,
            bool,
            "ScorerContract.requires_attestation",
        )
        if not self.evidence_action_kinds and not self.requires_attestation:
            raise ValueError(
                "A scorer contract needs evidence actions or an external attestation."
            )
        if self.kind is ScorerKind.HUMAN and not self.requires_attestation:
            raise ValueError("Human scorer contracts require an attestation.")

    @property
    def key(self) -> tuple[ScorerKind, str, str]:
        return (self.kind, self.scorer_id, self.scorer_version)

    def terms(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "scorer_id": self.scorer_id,
            "scorer_version": self.scorer_version,
            "criterion_ids": list(self.criterion_ids),
            "evidence_action_kinds": [
                kind.value for kind in self.evidence_action_kinds
            ],
            "check_set_ids": (
                None if self.check_set_ids is None else list(self.check_set_ids)
            ),
            "artifact_kinds": (
                None if self.artifact_kinds is None else list(self.artifact_kinds)
            ),
            "requires_attestation": self.requires_attestation,
        }


def _require_exact_type(value: object, expected: type, label: str) -> None:
    if type(value) is not expected:
        raise ValueError(f"{label} must be {expected.__name__}.")


def _require_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"{label} must match {_ID_PATTERN.pattern!r}; free-form text is not allowed."
        )
    return value


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not _DIGEST_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest.")
    return value


def _require_nonblank_text(value: object, label: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-blank string.")
    if len(value) > maximum:
        raise ValueError(f"{label} must contain at most {maximum} characters.")
    return value


def _finite_number(value: object, label: str) -> float:
    try:
        finite = type(value) in {int, float} and isfinite(float(value))
    except (OverflowError, ValueError):
        finite = False
    if not finite:
        raise ValueError(f"{label} must be a finite number.")
    return float(value)


def _unit_interval(value: object, label: str) -> float:
    number = _finite_number(value, label)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1 inclusive.")
    return number


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or not 0 <= value <= MAX_ACTION_COUNTER:
        raise ValueError(
            f"{label} must be an integer between 0 and {MAX_ACTION_COUNTER}."
        )
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or not 1 <= value <= MAX_ACTION_COUNTER:
        raise ValueError(
            f"{label} must be an integer between 1 and {MAX_ACTION_COUNTER}."
        )
    return value


def _require_enum(value: object, enum_type: type[StrEnum], label: str) -> None:
    if not isinstance(value, enum_type):
        raise ValueError(f"{label} must be a {enum_type.__name__} value.")


def _sorted_unique_ids(values: object, label: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise ValueError(f"{label} must be a tuple.")
    normalized = tuple(_require_id(value, f"{label}[]") for value in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} must not contain duplicates.")
    return tuple(sorted(normalized))


def _freeze_json(value: Any, path: str = "$") -> Any:
    """Validate finite JSON and return a deeply immutable representation."""

    if value is None or type(value) in {bool, str, int}:
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity.")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{path} object keys must be strings.")
        frozen: dict[str, Any] = {}
        for key in sorted(value):
            frozen[key] = _freeze_json(value[key], f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{path}[]") for item in value)
    raise ValueError(
        f"{path} must contain only finite JSON values; received {type(value).__name__}."
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Serialize a finite JSON value using TSQ's deterministic representation."""

    frozen = _freeze_json(value)
    return json.dumps(
        _thaw_json(frozen),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_digest(value: Any) -> str:
    """Return a SHA-256 digest of :func:`canonical_json`."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _exact_payload_keys(
    payload: Mapping[str, Any], required: set[str], kind: ActionKind
) -> None:
    actual = set(payload)
    if actual != required:
        missing = sorted(required - actual)
        unexpected = sorted(actual - required)
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ValueError(f"{kind.value} payload has " + "; ".join(details) + ".")


def _validate_action_payload(
    kind: ActionKind, payload: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Validate a closed action schema and return deeply immutable JSON."""

    _exact_payload_keys(payload, set(ACTION_PAYLOAD_CONTRACTS[kind]), kind)
    if kind is ActionKind.STARTED:
        pass
    elif kind is ActionKind.HINT_REQUESTED:
        _require_id(payload["hint_id"], "hint_id")
        _positive_int(payload["level"], "hint level")
    elif kind is ActionKind.ANSWER_REVISED:
        _require_digest(payload["answer_digest"], "answer_digest")
    elif kind is ActionKind.ARTIFACT_CHECKPOINT:
        _require_digest(payload["artifact_digest"], "artifact_digest")
        _require_id(payload["artifact_kind"], "artifact_kind")
    elif kind is ActionKind.EXPLANATION_CHECKPOINT:
        _require_digest(payload["explanation_digest"], "explanation_digest")
    elif kind is ActionKind.CHECK_RUN:
        _require_id(payload["check_set_id"], "check_set_id")
        for field_name in ("passed", "failed", "errored", "skipped"):
            _nonnegative_int(payload[field_name], field_name)
        _require_digest(payload["result_digest"], "result_digest")
    elif kind is ActionKind.TOOL_USED:
        _require_id(payload["tool_id"], "tool_id")
        _require_id(payload["purpose_code"], "purpose_code")
    elif kind is ActionKind.SUBMITTED:
        _require_digest(payload["submission_digest"], "submission_digest")
    elif kind is ActionKind.FEEDBACK_SHOWN:
        _require_digest(payload["feedback_digest"], "feedback_digest")
    elif kind is ActionKind.ABANDONED:
        _require_id(payload["reason_code"], "reason_code")
    else:  # pragma: no cover - exhaustive guard for future enum additions
        raise ValueError(f"No closed payload schema exists for {kind!r}.")
    return _freeze_json(payload, "$.payload")


@dataclass(frozen=True, slots=True)
class LearningAction:
    """One semantic action checkpoint in an immutable attempt trace.

    ``payload`` is copied into a deeply immutable mapping.  Each action kind has
    an exact schema containing only identifiers, counters, and SHA-256 digests.
    In particular, there is no field capable of accepting code, prose, a shell
    command, or tool output.
    """

    id: str
    trace_id: str
    sequence: int
    kind: ActionKind
    phase: ActionPhase
    payload: Mapping[str, Any] = field(default_factory=dict)
    elapsed_ms: int | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_id(self.id, "LearningAction.id")
        _require_id(self.trace_id, "LearningAction.trace_id")
        _nonnegative_int(self.sequence, "LearningAction.sequence")
        _require_enum(self.kind, ActionKind, "LearningAction.kind")
        _require_enum(self.phase, ActionPhase, "LearningAction.phase")
        if (
            self.kind is ActionKind.FEEDBACK_SHOWN
            and self.phase is not ActionPhase.POST_FEEDBACK
        ):
            raise ValueError("feedback_shown actions must be post_feedback.")
        if self.kind is ActionKind.STARTED and self.phase is not ActionPhase.UNASSISTED:
            raise ValueError("started actions must be unassisted.")
        if self.kind in {ActionKind.SUBMITTED, ActionKind.ABANDONED} and (
            self.phase is ActionPhase.POST_FEEDBACK
        ):
            raise ValueError(
                f"{self.kind.value} cannot be declared after feedback."
            )
        if not isinstance(self.payload, Mapping):
            raise ValueError("LearningAction.payload must be a mapping.")
        object.__setattr__(self, "payload", _validate_action_payload(self.kind, self.payload))
        if self.elapsed_ms is not None:
            _nonnegative_int(self.elapsed_ms, "LearningAction.elapsed_ms")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"LearningAction.schema_version must be {SCHEMA_VERSION}."
            )

    def terms(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "trace_id": self.trace_id,
            "sequence": self.sequence,
            "kind": self.kind.value,
            "phase": self.phase.value,
            "payload": _thaw_json(self.payload),
            "elapsed_ms": self.elapsed_ms,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class CheckProgress:
    """Safe aggregate for one deterministic check/test checkpoint."""

    action_id: str
    sequence: int
    check_set_id: str
    passed: int
    failed: int
    errored: int
    skipped: int
    pass_rate: float | None
    result_digest: str
    phase: ActionPhase

    def __post_init__(self) -> None:
        _require_id(self.action_id, "CheckProgress.action_id")
        _nonnegative_int(self.sequence, "CheckProgress.sequence")
        _require_id(self.check_set_id, "CheckProgress.check_set_id")
        for name in ("passed", "failed", "errored", "skipped"):
            _nonnegative_int(getattr(self, name), f"CheckProgress.{name}")
        if self.pass_rate is not None:
            object.__setattr__(
                self,
                "pass_rate",
                _unit_interval(self.pass_rate, "CheckProgress.pass_rate"),
            )
        _require_digest(self.result_digest, "CheckProgress.result_digest")
        _require_enum(self.phase, ActionPhase, "CheckProgress.phase")

    def terms(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "sequence": self.sequence,
            "check_set_id": self.check_set_id,
            "passed": self.passed,
            "failed": self.failed,
            "errored": self.errored,
            "skipped": self.skipped,
            "pass_rate": self.pass_rate,
            "result_digest": self.result_digest,
            "phase": self.phase.value,
        }


@dataclass(frozen=True, slots=True)
class ActionTraceSummary:
    """Deterministic, content-free summary derived from a semantic trace."""

    trace_id: str | None
    action_count: int
    elapsed_ms: int | None
    hint_count: int
    answer_revision_count: int
    tool_use_count: int
    check_run_count: int
    unassisted_action_count: int
    assisted_action_count: int
    post_feedback_action_count: int
    answer_digests: tuple[str, ...]
    artifact_digests: tuple[str, ...]
    explanation_digests: tuple[str, ...]
    submission_digests: tuple[str, ...]
    tool_ids: tuple[str, ...]
    test_progression: tuple[CheckProgress, ...]
    feedback_sequence: int | None
    abandoned: bool
    phase_correction_action_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.trace_id is not None:
            _require_id(self.trace_id, "ActionTraceSummary.trace_id")
        for name in (
            "action_count",
            "hint_count",
            "answer_revision_count",
            "tool_use_count",
            "check_run_count",
            "unassisted_action_count",
            "assisted_action_count",
            "post_feedback_action_count",
        ):
            _nonnegative_int(getattr(self, name), f"ActionTraceSummary.{name}")
        if self.elapsed_ms is not None:
            _nonnegative_int(self.elapsed_ms, "ActionTraceSummary.elapsed_ms")
        for name in (
            "answer_digests",
            "artifact_digests",
            "explanation_digests",
            "submission_digests",
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple):
                raise ValueError(f"ActionTraceSummary.{name} must be a tuple.")
            for value in values:
                _require_digest(value, f"ActionTraceSummary.{name}[]")
        object.__setattr__(
            self,
            "tool_ids",
            _sorted_unique_ids(self.tool_ids, "ActionTraceSummary.tool_ids"),
        )
        if not isinstance(self.test_progression, tuple) or any(
            not isinstance(item, CheckProgress) for item in self.test_progression
        ):
            raise ValueError(
                "ActionTraceSummary.test_progression must be a tuple of CheckProgress."
            )
        if self.feedback_sequence is not None:
            _nonnegative_int(
                self.feedback_sequence, "ActionTraceSummary.feedback_sequence"
            )
        _require_exact_type(self.abandoned, bool, "ActionTraceSummary.abandoned")
        object.__setattr__(
            self,
            "phase_correction_action_ids",
            _sorted_unique_ids(
                self.phase_correction_action_ids,
                "ActionTraceSummary.phase_correction_action_ids",
            ),
        )
        phase_total = (
            self.unassisted_action_count
            + self.assisted_action_count
            + self.post_feedback_action_count
        )
        if phase_total != self.action_count:
            raise ValueError("ActionTraceSummary phase counts must equal action_count.")
        if len(self.test_progression) != self.check_run_count:
            raise ValueError(
                "ActionTraceSummary.test_progression must match check_run_count."
            )

    @property
    def telemetry_present(self) -> bool:
        return self.action_count > 0

    @property
    def digest(self) -> str:
        return canonical_digest(self.terms())

    def terms(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "action_count": self.action_count,
            "elapsed_ms": self.elapsed_ms,
            "hint_count": self.hint_count,
            "answer_revision_count": self.answer_revision_count,
            "tool_use_count": self.tool_use_count,
            "check_run_count": self.check_run_count,
            "unassisted_action_count": self.unassisted_action_count,
            "assisted_action_count": self.assisted_action_count,
            "post_feedback_action_count": self.post_feedback_action_count,
            "answer_digests": list(self.answer_digests),
            "artifact_digests": list(self.artifact_digests),
            "explanation_digests": list(self.explanation_digests),
            "submission_digests": list(self.submission_digests),
            "tool_ids": list(self.tool_ids),
            "test_progression": [item.terms() for item in self.test_progression],
            "feedback_sequence": self.feedback_sequence,
            "abandoned": self.abandoned,
            "phase_correction_action_ids": list(self.phase_correction_action_ids),
        }


@dataclass(frozen=True, slots=True)
class RubricCriterion:
    """One observable rubric claim and its finite evidence budget.

    ``concept_weights`` is an explicit normalized mapping rather than an
    inference from prose.  ``dependence_group`` joins observations that share a
    stimulus, artifact, template, scorer, or other source of local dependence.
    All criteria in the same group must declare the same ``dependence_cap``.
    """

    id: str
    name: str
    scale: CriterionScale
    concept_weights: tuple[tuple[str, float], ...]
    dependence_group: str
    allowed_scores: tuple[float, ...] | None = None
    misconception_ids: tuple[str, ...] = ()
    score_weight: float = 1.0
    evidence_cap: float = 1.0
    dependence_cap: float = 1.0
    assisted_evidence_factor: float = 0.0
    certification_eligible: bool = True

    def __post_init__(self) -> None:
        _require_id(self.id, "RubricCriterion.id")
        _require_nonblank_text(self.name, "RubricCriterion.name", maximum=256)
        _require_enum(self.scale, CriterionScale, "RubricCriterion.scale")
        if self.scale is CriterionScale.CONTINUOUS:
            if self.allowed_scores is not None:
                raise ValueError(
                    "Continuous rubric criteria cannot declare discrete allowed_scores."
                )
        elif self.scale is CriterionScale.BINARY:
            if self.allowed_scores is not None and self.allowed_scores not in (
                (0.0, 1.0),
                (0, 1),
            ):
                raise ValueError("Binary rubric criteria allow only scores 0 and 1.")
            object.__setattr__(self, "allowed_scores", (0.0, 1.0))
        else:
            if not isinstance(self.allowed_scores, tuple) or len(self.allowed_scores) < 2:
                raise ValueError(
                    "Ordered and categorical rubric criteria require at least two allowed_scores."
                )
            normalized_scores = tuple(
                sorted(
                    {
                        _unit_interval(score, "RubricCriterion.allowed_scores[]")
                        for score in self.allowed_scores
                    }
                )
            )
            if len(normalized_scores) != len(self.allowed_scores):
                raise ValueError("RubricCriterion.allowed_scores must be unique.")
            object.__setattr__(self, "allowed_scores", normalized_scores)
        if not isinstance(self.concept_weights, tuple) or not self.concept_weights:
            raise ValueError("RubricCriterion.concept_weights must be a non-empty tuple.")
        seen: set[str] = set()
        normalized_weights: list[tuple[str, float]] = []
        for index, pair in enumerate(self.concept_weights):
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(
                    "RubricCriterion.concept_weights entries must be (concept_id, weight) tuples."
                )
            concept_id = _require_id(pair[0], f"concept_weights[{index}].concept_id")
            if concept_id in seen:
                raise ValueError(f"Duplicate concept mapping {concept_id}.")
            seen.add(concept_id)
            weight = _finite_number(pair[1], f"concept_weights[{index}].weight")
            if weight <= 0.0:
                raise ValueError("Concept weights must be positive.")
            normalized_weights.append((concept_id, weight))
        total = sum(weight for _, weight in normalized_weights)
        if abs(total - 1.0) > 1e-9:
            raise ValueError("RubricCriterion concept weights must sum to 1.")
        object.__setattr__(
            self,
            "concept_weights",
            tuple(sorted(normalized_weights, key=lambda pair: pair[0])),
        )
        _require_id(self.dependence_group, "RubricCriterion.dependence_group")
        object.__setattr__(
            self,
            "misconception_ids",
            _sorted_unique_ids(
                self.misconception_ids, "RubricCriterion.misconception_ids"
            ),
        )
        score_weight = _finite_number(self.score_weight, "RubricCriterion.score_weight")
        if score_weight <= 0.0:
            raise ValueError("RubricCriterion.score_weight must be positive.")
        object.__setattr__(self, "score_weight", score_weight)
        object.__setattr__(
            self,
            "evidence_cap",
            _unit_interval(self.evidence_cap, "RubricCriterion.evidence_cap"),
        )
        object.__setattr__(
            self,
            "dependence_cap",
            _unit_interval(self.dependence_cap, "RubricCriterion.dependence_cap"),
        )
        object.__setattr__(
            self,
            "assisted_evidence_factor",
            _unit_interval(
                self.assisted_evidence_factor,
                "RubricCriterion.assisted_evidence_factor",
            ),
        )
        _require_exact_type(
            self.certification_eligible,
            bool,
            "RubricCriterion.certification_eligible",
        )

    @property
    def concept_ids(self) -> tuple[str, ...]:
        return tuple(concept_id for concept_id, _ in self.concept_weights)

    def terms(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "scale": self.scale.value,
            "allowed_scores": (
                None if self.allowed_scores is None else list(self.allowed_scores)
            ),
            "concept_weights": [list(pair) for pair in self.concept_weights],
            "dependence_group": self.dependence_group,
            "misconception_ids": list(self.misconception_ids),
            "score_weight": self.score_weight,
            "evidence_cap": self.evidence_cap,
            "dependence_cap": self.dependence_cap,
            "assisted_evidence_factor": self.assisted_evidence_factor,
            "certification_eligible": self.certification_eligible,
        }


@dataclass(frozen=True, slots=True)
class LearningTask:
    """Immutable multimodal task contract consumed by the pure reducer."""

    id: str
    version: int
    family_id: str
    title: str
    modality: TaskModality
    criteria: tuple[RubricCriterion, ...]
    scorer_contracts: tuple[ScorerContract, ...] = ()
    allowed_action_kinds: tuple[ActionKind, ...] = tuple(ActionKind)
    allowed_tool_ids: tuple[str, ...] | None = None
    evidence_cap: float = 1.0
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_id(self.id, "LearningTask.id")
        _positive_int(self.version, "LearningTask.version")
        _require_id(self.family_id, "LearningTask.family_id")
        _require_nonblank_text(self.title, "LearningTask.title", maximum=256)
        _require_enum(self.modality, TaskModality, "LearningTask.modality")
        if not isinstance(self.criteria, tuple) or not self.criteria:
            raise ValueError("LearningTask.criteria must be a non-empty tuple.")
        if any(not isinstance(item, RubricCriterion) for item in self.criteria):
            raise ValueError("LearningTask.criteria must contain RubricCriterion values.")
        criterion_ids = [item.id for item in self.criteria]
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("LearningTask criterion IDs must be unique.")
        if not isinstance(self.scorer_contracts, tuple) or any(
            not isinstance(contract, ScorerContract)
            for contract in self.scorer_contracts
        ):
            raise ValueError(
                "LearningTask.scorer_contracts must contain ScorerContract values."
            )
        scorer_keys = [contract.key for contract in self.scorer_contracts]
        if len(scorer_keys) != len(set(scorer_keys)):
            raise ValueError("LearningTask scorer contracts must be unique.")
        object.__setattr__(
            self,
            "scorer_contracts",
            tuple(
                sorted(
                    self.scorer_contracts,
                    key=lambda contract: (
                        contract.kind.value,
                        contract.scorer_id,
                        contract.scorer_version,
                    ),
                )
            ),
        )
        group_caps: dict[str, float] = {}
        for criterion in self.criteria:
            previous = group_caps.setdefault(
                criterion.dependence_group, criterion.dependence_cap
            )
            if abs(previous - criterion.dependence_cap) > 1e-12:
                raise ValueError(
                    "Criteria in dependence group "
                    f"{criterion.dependence_group} must use the same cap."
                )
        if not isinstance(self.allowed_action_kinds, tuple):
            raise ValueError("LearningTask.allowed_action_kinds must be a tuple.")
        if any(not isinstance(kind, ActionKind) for kind in self.allowed_action_kinds):
            raise ValueError(
                "LearningTask.allowed_action_kinds must contain ActionKind values."
            )
        if len(self.allowed_action_kinds) != len(set(self.allowed_action_kinds)):
            raise ValueError("LearningTask.allowed_action_kinds must be unique.")
        object.__setattr__(
            self,
            "allowed_action_kinds",
            tuple(sorted(self.allowed_action_kinds, key=lambda item: item.value)),
        )
        allowed_kinds = set(self.allowed_action_kinds)
        for contract in self.scorer_contracts:
            unknown_criteria = set(contract.criterion_ids) - set(criterion_ids)
            if unknown_criteria:
                raise ValueError(
                    f"Scorer {contract.scorer_id} references unknown criteria: "
                    + ", ".join(sorted(unknown_criteria))
                    + "."
                )
            disallowed = set(contract.evidence_action_kinds) - allowed_kinds
            if disallowed:
                raise ValueError(
                    f"Scorer {contract.scorer_id} references task-disallowed actions: "
                    + ", ".join(sorted(kind.value for kind in disallowed))
                    + "."
                )
        if self.allowed_tool_ids is not None:
            object.__setattr__(
                self,
                "allowed_tool_ids",
                _sorted_unique_ids(
                    self.allowed_tool_ids, "LearningTask.allowed_tool_ids"
                ),
            )
        object.__setattr__(
            self,
            "evidence_cap",
            _unit_interval(self.evidence_cap, "LearningTask.evidence_cap"),
        )
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"LearningTask.schema_version must be {SCHEMA_VERSION}.")

    @property
    def concept_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    concept_id
                    for criterion in self.criteria
                    for concept_id in criterion.concept_ids
                }
            )
        )

    @property
    def misconception_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    misconception_id
                    for criterion in self.criteria
                    for misconception_id in criterion.misconception_ids
                }
            )
        )

    @property
    def digest(self) -> str:
        return canonical_digest(self.terms())

    def terms(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "family_id": self.family_id,
            "title": self.title,
            "modality": self.modality.value,
            "criteria": [criterion.terms() for criterion in self.criteria],
            "scorer_contracts": [
                contract.terms() for contract in self.scorer_contracts
            ],
            "allowed_action_kinds": [kind.value for kind in self.allowed_action_kinds],
            "allowed_tool_ids": (
                None if self.allowed_tool_ids is None else list(self.allowed_tool_ids)
            ),
            "evidence_cap": self.evidence_cap,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class CriterionEvaluation:
    """A scorer's normalized observation for one rubric criterion.

    An invalid or missing observation has ``score=None``.  ``MODEL`` and raw
    ``IMPORTED`` scores are retained for shadow comparison but receive no
    competence-evidence weight; an explicit human or deterministic evaluation
    is required.
    """

    criterion_id: str
    status: EvaluationStatus
    score: float | None
    outcome_code: str
    phase: ActionPhase
    scorer_kind: ScorerKind
    scorer_id: str
    scorer_version: str
    source_action_ids: tuple[str, ...] = ()
    attestation_digest: str | None = None
    misconception_ids: tuple[str, ...] = ()
    reliability: float = 1.0

    def __post_init__(self) -> None:
        _require_id(self.criterion_id, "CriterionEvaluation.criterion_id")
        _require_enum(self.status, EvaluationStatus, "CriterionEvaluation.status")
        if self.status is EvaluationStatus.VALID:
            if self.score is None:
                raise ValueError("A valid CriterionEvaluation requires a score.")
            object.__setattr__(
                self,
                "score",
                _unit_interval(self.score, "CriterionEvaluation.score"),
            )
        elif self.score is not None:
            raise ValueError("An invalid or missing CriterionEvaluation cannot carry a score.")
        _require_id(self.outcome_code, "CriterionEvaluation.outcome_code")
        _require_enum(self.phase, ActionPhase, "CriterionEvaluation.phase")
        _require_enum(self.scorer_kind, ScorerKind, "CriterionEvaluation.scorer_kind")
        _require_id(self.scorer_id, "CriterionEvaluation.scorer_id")
        _require_id(self.scorer_version, "CriterionEvaluation.scorer_version")
        object.__setattr__(
            self,
            "source_action_ids",
            _sorted_unique_ids(
                self.source_action_ids, "CriterionEvaluation.source_action_ids"
            ),
        )
        if self.attestation_digest is not None:
            _require_digest(
                self.attestation_digest,
                "CriterionEvaluation.attestation_digest",
            )
        object.__setattr__(
            self,
            "misconception_ids",
            _sorted_unique_ids(
                self.misconception_ids, "CriterionEvaluation.misconception_ids"
            ),
        )
        if self.score == 1.0 and self.misconception_ids:
            raise ValueError(
                "A fully successful criterion cannot assert an observed misconception."
            )
        object.__setattr__(
            self,
            "reliability",
            _unit_interval(self.reliability, "CriterionEvaluation.reliability"),
        )

    def terms(self) -> dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "status": self.status.value,
            "score": self.score,
            "outcome_code": self.outcome_code,
            "phase": self.phase.value,
            "scorer_kind": self.scorer_kind.value,
            "scorer_id": self.scorer_id,
            "scorer_version": self.scorer_version,
            "source_action_ids": list(self.source_action_ids),
            "attestation_digest": self.attestation_digest,
            "misconception_ids": list(self.misconception_ids),
            "reliability": self.reliability,
        }


@dataclass(frozen=True, slots=True)
class TaskEvaluation:
    """Version-pinned set of criterion observations for one task attempt."""

    id: str
    trace_id: str
    task_id: str
    task_version: int
    task_digest: str
    action_trace_digest: str
    criteria: tuple[CriterionEvaluation, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_id(self.id, "TaskEvaluation.id")
        _require_id(self.trace_id, "TaskEvaluation.trace_id")
        _require_id(self.task_id, "TaskEvaluation.task_id")
        _positive_int(self.task_version, "TaskEvaluation.task_version")
        _require_digest(self.task_digest, "TaskEvaluation.task_digest")
        _require_digest(
            self.action_trace_digest,
            "TaskEvaluation.action_trace_digest",
        )
        if not isinstance(self.criteria, tuple) or any(
            not isinstance(item, CriterionEvaluation) for item in self.criteria
        ):
            raise ValueError(
                "TaskEvaluation.criteria must be a tuple of CriterionEvaluation values."
            )
        criterion_ids = [item.criterion_id for item in self.criteria]
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("TaskEvaluation criterion IDs must be unique.")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"TaskEvaluation.schema_version must be {SCHEMA_VERSION}.")

    @property
    def digest(self) -> str:
        return canonical_digest(self.terms())

    def terms(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "task_version": self.task_version,
            "task_digest": self.task_digest,
            "action_trace_digest": self.action_trace_digest,
            "criteria": [criterion.terms() for criterion in self.criteria],
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """Auditable evidence projection for one task rubric criterion."""

    id: str
    criterion_id: str
    score: float | None
    outcome_code: str
    concept_weights: tuple[tuple[str, float], ...]
    misconception_ids: tuple[str, ...]
    family_id: str
    dependence_group: str
    potential_weight: float
    requested_weight: float
    effective_weight: float
    certification_eligible: bool
    phase: ActionPhase
    status: EvaluationStatus
    scorer_kind: ScorerKind
    scorer_id: str
    scorer_version: str
    source_action_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    provenance_digest: str

    def __post_init__(self) -> None:
        _require_id(self.id, "EvidenceRecord.id")
        _require_id(self.criterion_id, "EvidenceRecord.criterion_id")
        if self.score is not None:
            object.__setattr__(
                self, "score", _unit_interval(self.score, "EvidenceRecord.score")
            )
        _require_id(self.outcome_code, "EvidenceRecord.outcome_code")
        if not isinstance(self.concept_weights, tuple) or not self.concept_weights:
            raise ValueError("EvidenceRecord.concept_weights must be non-empty.")
        concept_ids: set[str] = set()
        concept_total = 0.0
        for concept_id, weight in self.concept_weights:
            _require_id(concept_id, "EvidenceRecord.concept_id")
            if concept_id in concept_ids:
                raise ValueError("EvidenceRecord concept mappings must be unique.")
            concept_ids.add(concept_id)
            normalized_weight = _unit_interval(
                weight, "EvidenceRecord.concept_weight"
            )
            if normalized_weight <= 0.0:
                raise ValueError("EvidenceRecord concept weights must be positive.")
            concept_total += normalized_weight
        if abs(concept_total - 1.0) > 1e-9:
            raise ValueError("EvidenceRecord concept weights must sum to 1.")
        object.__setattr__(
            self,
            "misconception_ids",
            _sorted_unique_ids(
                self.misconception_ids, "EvidenceRecord.misconception_ids"
            ),
        )
        _require_id(self.family_id, "EvidenceRecord.family_id")
        _require_id(self.dependence_group, "EvidenceRecord.dependence_group")
        for name in ("potential_weight", "requested_weight", "effective_weight"):
            object.__setattr__(
                self,
                name,
                _unit_interval(getattr(self, name), f"EvidenceRecord.{name}"),
            )
        if self.requested_weight > self.potential_weight + 1e-12:
            raise ValueError("EvidenceRecord.requested_weight exceeds potential_weight.")
        if self.effective_weight > self.requested_weight + 1e-12:
            raise ValueError("EvidenceRecord.effective_weight exceeds requested_weight.")
        _require_exact_type(
            self.certification_eligible,
            bool,
            "EvidenceRecord.certification_eligible",
        )
        if self.certification_eligible and self.effective_weight <= 0.0:
            raise ValueError(
                "A certification-eligible EvidenceRecord must have positive weight."
            )
        _require_enum(self.phase, ActionPhase, "EvidenceRecord.phase")
        _require_enum(self.status, EvaluationStatus, "EvidenceRecord.status")
        _require_enum(self.scorer_kind, ScorerKind, "EvidenceRecord.scorer_kind")
        _require_id(self.scorer_id, "EvidenceRecord.scorer_id")
        _require_id(self.scorer_version, "EvidenceRecord.scorer_version")
        object.__setattr__(
            self,
            "source_action_ids",
            _sorted_unique_ids(
                self.source_action_ids, "EvidenceRecord.source_action_ids"
            ),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _sorted_unique_ids(self.reason_codes, "EvidenceRecord.reason_codes"),
        )
        _require_digest(self.provenance_digest, "EvidenceRecord.provenance_digest")

    def terms(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "criterion_id": self.criterion_id,
            "score": self.score,
            "outcome_code": self.outcome_code,
            "concept_weights": [list(pair) for pair in self.concept_weights],
            "misconception_ids": list(self.misconception_ids),
            "family_id": self.family_id,
            "dependence_group": self.dependence_group,
            "potential_weight": self.potential_weight,
            "requested_weight": self.requested_weight,
            "effective_weight": self.effective_weight,
            "certification_eligible": self.certification_eligible,
            "phase": self.phase.value,
            "status": self.status.value,
            "scorer_kind": self.scorer_kind.value,
            "scorer_id": self.scorer_id,
            "scorer_version": self.scorer_version,
            "source_action_ids": list(self.source_action_ids),
            "reason_codes": list(self.reason_codes),
            "provenance_digest": self.provenance_digest,
        }


@dataclass(frozen=True, slots=True)
class EvidenceGroupSummary:
    """Finite information budget consumed by one dependence group."""

    group_id: str
    cap: float
    requested_weight: float
    group_capped_weight: float
    effective_weight: float

    def __post_init__(self) -> None:
        _require_id(self.group_id, "EvidenceGroupSummary.group_id")
        object.__setattr__(
            self, "cap", _unit_interval(self.cap, "EvidenceGroupSummary.cap")
        )
        for name in ("requested_weight", "group_capped_weight", "effective_weight"):
            value = _finite_number(getattr(self, name), f"EvidenceGroupSummary.{name}")
            if value < 0.0:
                raise ValueError(f"EvidenceGroupSummary.{name} must be non-negative.")
            object.__setattr__(self, name, value)
        if self.group_capped_weight > self.cap + 1e-12:
            raise ValueError("Dependence-group weight exceeds its cap.")
        if self.group_capped_weight > self.requested_weight + 1e-12:
            raise ValueError("Capped group weight exceeds requested weight.")
        if self.effective_weight > self.group_capped_weight + 1e-12:
            raise ValueError("Effective group weight exceeds group-capped weight.")

    def terms(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "cap": self.cap,
            "requested_weight": self.requested_weight,
            "group_capped_weight": self.group_capped_weight,
            "effective_weight": self.effective_weight,
        }


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """Complete deterministic reduction of one task attempt."""

    task_id: str
    task_version: int
    task_digest: str
    trace_id: str
    evaluation_id: str
    evaluation_digest: str
    action_trace_digest: str
    trace: ActionTraceSummary
    records: tuple[EvidenceRecord, ...]
    groups: tuple[EvidenceGroupSummary, ...]
    reported_task_score: float | None
    evidence_score: float | None
    total_evidence_weight: float
    certification_evidence_weight: float
    missing_criterion_ids: tuple[str, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_id(self.task_id, "EvidenceBundle.task_id")
        _positive_int(self.task_version, "EvidenceBundle.task_version")
        _require_digest(self.task_digest, "EvidenceBundle.task_digest")
        _require_id(self.trace_id, "EvidenceBundle.trace_id")
        _require_id(self.evaluation_id, "EvidenceBundle.evaluation_id")
        _require_digest(self.evaluation_digest, "EvidenceBundle.evaluation_digest")
        _require_digest(
            self.action_trace_digest, "EvidenceBundle.action_trace_digest"
        )
        if not isinstance(self.trace, ActionTraceSummary):
            raise ValueError("EvidenceBundle.trace must be an ActionTraceSummary.")
        if not isinstance(self.records, tuple) or any(
            not isinstance(record, EvidenceRecord) for record in self.records
        ):
            raise ValueError("EvidenceBundle.records must contain EvidenceRecord values.")
        if not isinstance(self.groups, tuple) or any(
            not isinstance(group, EvidenceGroupSummary) for group in self.groups
        ):
            raise ValueError(
                "EvidenceBundle.groups must contain EvidenceGroupSummary values."
            )
        for name in ("reported_task_score", "evidence_score"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self, name, _unit_interval(value, f"EvidenceBundle.{name}")
                )
        for name in ("total_evidence_weight", "certification_evidence_weight"):
            value = _finite_number(getattr(self, name), f"EvidenceBundle.{name}")
            if value < 0.0:
                raise ValueError(f"EvidenceBundle.{name} must be non-negative.")
            object.__setattr__(self, name, value)
        if self.certification_evidence_weight > self.total_evidence_weight + 1e-12:
            raise ValueError(
                "EvidenceBundle certification weight exceeds total evidence weight."
            )
        object.__setattr__(
            self,
            "missing_criterion_ids",
            _sorted_unique_ids(
                self.missing_criterion_ids, "EvidenceBundle.missing_criterion_ids"
            ),
        )
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"EvidenceBundle.schema_version must be {SCHEMA_VERSION}.")

    @property
    def digest(self) -> str:
        return canonical_digest(self.terms())

    def terms(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_version": self.task_version,
            "task_digest": self.task_digest,
            "trace_id": self.trace_id,
            "evaluation_id": self.evaluation_id,
            "evaluation_digest": self.evaluation_digest,
            "action_trace_digest": self.action_trace_digest,
            "trace": self.trace.terms(),
            "records": [record.terms() for record in self.records],
            "groups": [group.terms() for group in self.groups],
            "reported_task_score": self.reported_task_score,
            "evidence_score": self.evidence_score,
            "total_evidence_weight": self.total_evidence_weight,
            "certification_evidence_weight": self.certification_evidence_weight,
            "missing_criterion_ids": list(self.missing_criterion_ids),
            "schema_version": self.schema_version,
        }


_PHASE_RANK = {
    ActionPhase.UNASSISTED: 0,
    ActionPhase.ASSISTED: 1,
    ActionPhase.POST_FEEDBACK: 2,
}
_RANK_PHASE = {rank: phase for phase, rank in _PHASE_RANK.items()}


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _summarize_actions(
    actions: Iterable[LearningAction],
) -> tuple[ActionTraceSummary, dict[str, ActionPhase]]:
    action_tuple = tuple(actions)
    if any(not isinstance(action, LearningAction) for action in action_tuple):
        raise ValueError("actions must contain only LearningAction values.")
    ordered = tuple(sorted(action_tuple, key=lambda item: (item.sequence, item.id)))
    ids = [action.id for action in ordered]
    sequences = [action.sequence for action in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("LearningAction IDs must be unique within a trace.")
    if len(sequences) != len(set(sequences)):
        raise ValueError("LearningAction sequences must be unique within a trace.")
    if sequences:
        first_sequence = sequences[0]
        if first_sequence not in {0, 1} or sequences != list(
            range(first_sequence, first_sequence + len(sequences))
        ):
            raise ValueError(
                "LearningAction sequences must be contiguous and start at zero or one."
            )
    singleton_kinds = {
        ActionKind.STARTED,
        ActionKind.SUBMITTED,
        ActionKind.ABANDONED,
        ActionKind.FEEDBACK_SHOWN,
    }
    kind_counts = {
        kind: sum(action.kind is kind for action in ordered)
        for kind in singleton_kinds
    }
    duplicated = sorted(kind.value for kind, count in kind_counts.items() if count > 1)
    if duplicated:
        raise ValueError(
            "A trace cannot repeat singleton lifecycle actions: "
            + ", ".join(duplicated)
            + "."
        )
    if kind_counts[ActionKind.STARTED] and ordered[0].kind is not ActionKind.STARTED:
        raise ValueError("A started action must be the first trace checkpoint.")
    if kind_counts[ActionKind.SUBMITTED] and kind_counts[ActionKind.ABANDONED]:
        raise ValueError("A trace cannot be both submitted and abandoned.")
    submitted_index = next(
        (
            index
            for index, action in enumerate(ordered)
            if action.kind is ActionKind.SUBMITTED
        ),
        None,
    )
    feedback_index = next(
        (
            index
            for index, action in enumerate(ordered)
            if action.kind is ActionKind.FEEDBACK_SHOWN
        ),
        None,
    )
    if (
        submitted_index is not None
        and feedback_index is not None
        and feedback_index < submitted_index
    ):
        raise ValueError("A feedback checkpoint cannot precede submission.")
    if submitted_index is not None and any(
        action.phase is not ActionPhase.POST_FEEDBACK
        for action in ordered[submitted_index + 1 :]
    ):
        raise ValueError(
            "Only post-feedback actions may follow a submitted checkpoint."
        )
    if kind_counts[ActionKind.ABANDONED] and ordered[-1].kind is not ActionKind.ABANDONED:
        raise ValueError("No action may follow an abandoned trace.")
    elapsed_sequence = [
        action.elapsed_ms for action in ordered if action.elapsed_ms is not None
    ]
    if any(
        current < prior
        for prior, current in zip(elapsed_sequence, elapsed_sequence[1:])
    ):
        raise ValueError("LearningAction elapsed times must be monotonic.")
    trace_ids = {action.trace_id for action in ordered}
    if len(trace_ids) > 1:
        raise ValueError("A trace cannot combine multiple trace IDs.")

    current_rank = 0
    effective_phases: dict[str, ActionPhase] = {}
    correction_ids: list[str] = []
    answer_digests: list[str] = []
    artifact_digests: list[str] = []
    explanation_digests: list[str] = []
    submission_digests: list[str] = []
    tool_ids: list[str] = []
    progression: list[CheckProgress] = []
    counts = {phase: 0 for phase in ActionPhase}
    hint_count = 0
    revision_count = 0
    tool_count = 0
    feedback_sequence: int | None = None
    abandoned = False

    for action in ordered:
        declared_rank = _PHASE_RANK[action.phase]
        effective_rank = max(current_rank, declared_rank)
        effective_phase = _RANK_PHASE[effective_rank]
        effective_phases[action.id] = effective_phase
        if effective_phase is not action.phase:
            correction_ids.append(action.id)
        counts[effective_phase] += 1

        if action.kind is ActionKind.HINT_REQUESTED:
            hint_count += 1
            current_rank = max(current_rank, 1)
        elif action.kind is ActionKind.ANSWER_REVISED:
            revision_count += 1
            _append_unique(answer_digests, action.payload["answer_digest"])
        elif action.kind is ActionKind.ARTIFACT_CHECKPOINT:
            _append_unique(artifact_digests, action.payload["artifact_digest"])
        elif action.kind is ActionKind.EXPLANATION_CHECKPOINT:
            _append_unique(
                explanation_digests, action.payload["explanation_digest"]
            )
        elif action.kind is ActionKind.CHECK_RUN:
            attempted = (
                action.payload["passed"]
                + action.payload["failed"]
                + action.payload["errored"]
            )
            pass_rate = (
                None if attempted == 0 else action.payload["passed"] / attempted
            )
            progression.append(
                CheckProgress(
                    action_id=action.id,
                    sequence=action.sequence,
                    check_set_id=action.payload["check_set_id"],
                    passed=action.payload["passed"],
                    failed=action.payload["failed"],
                    errored=action.payload["errored"],
                    skipped=action.payload["skipped"],
                    pass_rate=pass_rate,
                    result_digest=action.payload["result_digest"],
                    phase=effective_phase,
                )
            )
        elif action.kind is ActionKind.TOOL_USED:
            tool_count += 1
            _append_unique(tool_ids, action.payload["tool_id"])
        elif action.kind is ActionKind.SUBMITTED:
            _append_unique(submission_digests, action.payload["submission_digest"])
        elif action.kind is ActionKind.FEEDBACK_SHOWN:
            if feedback_sequence is None:
                feedback_sequence = action.sequence
            current_rank = 2
        elif action.kind is ActionKind.ABANDONED:
            abandoned = True

        current_rank = max(current_rank, declared_rank)

    elapsed_values = [
        action.elapsed_ms for action in ordered if action.elapsed_ms is not None
    ]
    summary = ActionTraceSummary(
        trace_id=next(iter(trace_ids)) if trace_ids else None,
        action_count=len(ordered),
        elapsed_ms=max(elapsed_values) if elapsed_values else None,
        hint_count=hint_count,
        answer_revision_count=revision_count,
        tool_use_count=tool_count,
        check_run_count=len(progression),
        unassisted_action_count=counts[ActionPhase.UNASSISTED],
        assisted_action_count=counts[ActionPhase.ASSISTED],
        post_feedback_action_count=counts[ActionPhase.POST_FEEDBACK],
        answer_digests=tuple(answer_digests),
        artifact_digests=tuple(artifact_digests),
        explanation_digests=tuple(explanation_digests),
        submission_digests=tuple(submission_digests),
        tool_ids=tuple(tool_ids),
        test_progression=tuple(progression),
        feedback_sequence=feedback_sequence,
        abandoned=abandoned,
        phase_correction_action_ids=tuple(correction_ids),
    )
    return summary, effective_phases


def summarize_actions(actions: Iterable[LearningAction]) -> ActionTraceSummary:
    """Return a deterministic, content-free summary of an action trace."""

    summary, _ = _summarize_actions(actions)
    return summary


def _action_trace_commitment_terms(
    actions: tuple[LearningAction, ...],
) -> dict[str, Any]:
    ordered = tuple(sorted(actions, key=lambda item: (item.sequence, item.id)))
    return {
        "type": "tsq.learning_action_trace",
        "schema_version": SCHEMA_VERSION,
        "actions": [action.terms() for action in ordered],
    }


def action_trace_digest(actions: Iterable[LearningAction]) -> str:
    """Commit to every typed field in a validated, canonical action trace.

    Unlike :attr:`ActionTraceSummary.digest`, this digest is not a projection:
    it binds action identity, trace identity, sequence, kind, declared phase,
    the complete closed payload, timing, and action schema version.  Iterable
    input order is irrelevant because the committed order is the validated
    semantic sequence.
    """

    action_tuple = tuple(actions)
    _summarize_actions(action_tuple)
    return canonical_digest(_action_trace_commitment_terms(action_tuple))


@dataclass(frozen=True, slots=True)
class _PendingRecord:
    criterion: RubricCriterion
    evaluation: CriterionEvaluation
    phase: ActionPhase
    potential_weight: float
    requested_weight: float
    reasons: tuple[str, ...]


class DeterministicEvidenceReducer:
    """Compile a task, scorer output, and semantic trace into capped evidence.

    The reducer performs no I/O and invokes no user-provided functions.  Invalid
    structural input raises ``ValueError``.  An invalid, missing, assisted, or
    post-feedback observation remains visible in the output with zero or capped
    weight instead of being converted into negative learner evidence.
    """

    def reduce(
        self,
        task: LearningTask,
        evaluation: TaskEvaluation,
        actions: Iterable[LearningAction] = (),
    ) -> EvidenceBundle:
        if not isinstance(task, LearningTask):
            raise ValueError("task must be a LearningTask.")
        if not isinstance(evaluation, TaskEvaluation):
            raise ValueError("evaluation must be a TaskEvaluation.")
        if (
            evaluation.task_id != task.id
            or evaluation.task_version != task.version
            or evaluation.task_digest != task.digest
        ):
            raise ValueError("TaskEvaluation is not pinned to the supplied LearningTask.")

        action_tuple = tuple(actions)
        trace, effective_phases = _summarize_actions(action_tuple)
        full_trace_digest = canonical_digest(
            _action_trace_commitment_terms(action_tuple)
        )
        if trace.trace_id is not None and trace.trace_id != evaluation.trace_id:
            raise ValueError("Action trace identity does not match TaskEvaluation.")
        if evaluation.action_trace_digest != full_trace_digest:
            raise ValueError(
                "TaskEvaluation is not pinned to the supplied semantic action trace."
            )
        disallowed = sorted(
            {
                action.kind.value
                for action in action_tuple
                if action.kind not in task.allowed_action_kinds
            }
        )
        if disallowed:
            raise ValueError(
                "Action trace violates the task telemetry contract: "
                + ", ".join(disallowed)
                + "."
            )

        action_by_id = {action.id: action for action in action_tuple}
        scorer_contracts = {
            contract.key: contract for contract in task.scorer_contracts
        }
        unallowed_tools: set[str] = set()
        if task.allowed_tool_ids is not None:
            allowed_tools = set(task.allowed_tool_ids)
            closure_sequences = [
                action.sequence
                for action in action_tuple
                if action.kind in {ActionKind.SUBMITTED, ActionKind.FEEDBACK_SHOWN}
            ]
            administration_end = min(closure_sequences, default=None)
            unallowed_tools = {
                action.payload["tool_id"]
                for action in action_tuple
                if action.kind is ActionKind.TOOL_USED
                and action.payload["tool_id"] not in allowed_tools
                and (
                    administration_end is None
                    or action.sequence < administration_end
                )
            }

        evaluation_by_criterion = {
            item.criterion_id: item for item in evaluation.criteria
        }
        task_criterion_ids = {criterion.id for criterion in task.criteria}
        unknown_evaluations = sorted(set(evaluation_by_criterion) - task_criterion_ids)
        if unknown_evaluations:
            raise ValueError(
                "TaskEvaluation contains unknown rubric criteria: "
                + ", ".join(unknown_evaluations)
                + "."
            )

        pending: list[_PendingRecord] = []
        missing: list[str] = []
        for criterion in task.criteria:
            observed = evaluation_by_criterion.get(criterion.id)
            if observed is None:
                missing.append(criterion.id)
                observed = CriterionEvaluation(
                    criterion_id=criterion.id,
                    status=EvaluationStatus.MISSING,
                    score=None,
                    outcome_code="missing",
                    phase=ActionPhase.UNASSISTED,
                    scorer_kind=ScorerKind.IMPORTED,
                    scorer_id="none",
                    scorer_version="none",
                )
            unexpected_misconceptions = set(observed.misconception_ids) - set(
                criterion.misconception_ids
            )
            if unexpected_misconceptions:
                raise ValueError(
                    f"Criterion {criterion.id} reported undeclared misconceptions: "
                    + ", ".join(sorted(unexpected_misconceptions))
                    + "."
                )
            if (
                observed.status is EvaluationStatus.VALID
                and observed.score is not None
                and criterion.allowed_scores is not None
                and not any(
                    abs(observed.score - allowed) <= 1e-12
                    for allowed in criterion.allowed_scores
                )
            ):
                raise ValueError(
                    f"Criterion {criterion.id} score is not permitted by its "
                    f"{criterion.scale.value} scale."
                )

            reasons: list[str] = []
            effective_phase = observed.phase
            missing_sources = [
                source_id
                for source_id in observed.source_action_ids
                if source_id not in action_by_id
            ]
            if missing_sources:
                reasons.append("missing_source_action")
            source_phases = [
                effective_phases[source_id]
                for source_id in observed.source_action_ids
                if source_id in effective_phases
            ]
            if source_phases:
                source_phase = max(source_phases, key=lambda item: _PHASE_RANK[item])
                if _PHASE_RANK[source_phase] > _PHASE_RANK[effective_phase]:
                    effective_phase = source_phase
                    reasons.append("source_phase_escalated")

            potential = criterion.evidence_cap * observed.reliability
            requested = potential
            if observed.status is EvaluationStatus.INVALID:
                requested = 0.0
                reasons.append("invalid_evaluation")
            elif observed.status is EvaluationStatus.MISSING:
                requested = 0.0
                reasons.append("missing_evaluation")
            if missing_sources:
                requested = 0.0
            if observed.scorer_kind in {
                ScorerKind.DETERMINISTIC,
                ScorerKind.HUMAN,
            }:
                contract = scorer_contracts.get(
                    (
                        observed.scorer_kind,
                        observed.scorer_id,
                        observed.scorer_version,
                    )
                )
                if contract is None:
                    requested = 0.0
                    reasons.append("untrusted_scorer")
                else:
                    if criterion.id not in contract.criterion_ids:
                        requested = 0.0
                        reasons.append("scorer_not_authorized_for_criterion")
                    existing_sources = [
                        action_by_id[source_id]
                        for source_id in observed.source_action_ids
                        if source_id in action_by_id
                    ]
                    if existing_sources and any(
                        source.kind not in contract.evidence_action_kinds
                        for source in existing_sources
                    ):
                        requested = 0.0
                        reasons.append("non_evidence_source_action")
                    if (
                        contract.check_set_ids is not None
                        and any(
                            source.kind is ActionKind.CHECK_RUN
                            and source.payload["check_set_id"]
                            not in contract.check_set_ids
                            for source in existing_sources
                        )
                    ):
                        requested = 0.0
                        reasons.append("unauthorized_check_set")
                    if (
                        contract.artifact_kinds is not None
                        and any(
                            source.kind is ActionKind.ARTIFACT_CHECKPOINT
                            and source.payload["artifact_kind"]
                            not in contract.artifact_kinds
                            for source in existing_sources
                        )
                    ):
                        requested = 0.0
                        reasons.append("unauthorized_artifact_kind")
                    if (
                        not observed.source_action_ids
                        and not contract.requires_attestation
                    ):
                        requested = 0.0
                        reasons.append("missing_evidence_source")
                    if (
                        contract.requires_attestation
                        and observed.attestation_digest is None
                    ):
                        requested = 0.0
                        reasons.append("missing_scorer_attestation")
            if effective_phase is ActionPhase.ASSISTED:
                requested *= criterion.assisted_evidence_factor
                reasons.append("assisted_observation")
            elif effective_phase is ActionPhase.POST_FEEDBACK:
                requested = 0.0
                reasons.append("post_feedback_observation")
            if observed.scorer_kind is ScorerKind.MODEL:
                requested = 0.0
                reasons.append("model_score_shadow_only")
            elif observed.scorer_kind is ScorerKind.IMPORTED:
                requested = 0.0
                reasons.append("imported_score_unadjudicated")
            if unallowed_tools:
                # Tool use is not evidence of low competence.  It invalidates a
                # restricted administration condition, so the observation is
                # retained with zero weight instead of becoming a penalty.
                requested = 0.0
                reasons.append("task_condition_invalid")
            if not reasons:
                reasons.append("valid_observation")

            pending.append(
                _PendingRecord(
                    criterion=criterion,
                    evaluation=observed,
                    phase=effective_phase,
                    potential_weight=potential,
                    requested_weight=requested,
                    reasons=tuple(sorted(set(reasons))),
                )
            )

        requested_by_group: dict[str, float] = {}
        cap_by_group: dict[str, float] = {}
        for item in pending:
            group = item.criterion.dependence_group
            requested_by_group[group] = (
                requested_by_group.get(group, 0.0) + item.requested_weight
            )
            cap_by_group[group] = item.criterion.dependence_cap
        group_scales = {
            group: (
                1.0
                if requested <= cap_by_group[group] or requested <= 0.0
                else cap_by_group[group] / requested
            )
            for group, requested in requested_by_group.items()
        }
        group_capped_total = sum(
            requested_by_group[group] * scale
            for group, scale in group_scales.items()
        )
        task_scale = (
            1.0
            if group_capped_total <= task.evidence_cap or group_capped_total <= 0.0
            else task.evidence_cap / group_capped_total
        )

        records: list[EvidenceRecord] = []
        for item in pending:
            group_scale = group_scales[item.criterion.dependence_group]
            effective_weight = item.requested_weight * group_scale * task_scale
            reasons = list(item.reasons)
            if group_scale < 1.0:
                reasons.append("dependence_group_capped")
            if task_scale < 1.0:
                reasons.append("task_evidence_capped")
            certification_eligible = (
                item.criterion.certification_eligible
                and item.phase is ActionPhase.UNASSISTED
                and item.evaluation.status is EvaluationStatus.VALID
                and item.evaluation.scorer_kind
                in {ScorerKind.DETERMINISTIC, ScorerKind.HUMAN}
                and not unallowed_tools
                and effective_weight > 0.0
            )
            provenance = {
                "task_digest": task.digest,
                "evaluation_digest": evaluation.digest,
                "action_trace_digest": full_trace_digest,
                "trace_summary_digest": trace.digest,
                "criterion_id": item.criterion.id,
                "effective_phase": item.phase.value,
                "reducer": "deterministic-evidence-v1",
            }
            record_material = {
                **provenance,
                "trace_id": evaluation.trace_id,
            }
            records.append(
                EvidenceRecord(
                    id="evr_" + canonical_digest(record_material)[:24],
                    criterion_id=item.criterion.id,
                    score=item.evaluation.score,
                    outcome_code=item.evaluation.outcome_code,
                    concept_weights=item.criterion.concept_weights,
                    misconception_ids=item.evaluation.misconception_ids,
                    family_id=task.family_id,
                    dependence_group=item.criterion.dependence_group,
                    potential_weight=item.potential_weight,
                    requested_weight=item.requested_weight,
                    effective_weight=effective_weight,
                    certification_eligible=certification_eligible,
                    phase=item.phase,
                    status=item.evaluation.status,
                    scorer_kind=item.evaluation.scorer_kind,
                    scorer_id=item.evaluation.scorer_id,
                    scorer_version=item.evaluation.scorer_version,
                    source_action_ids=item.evaluation.source_action_ids,
                    reason_codes=tuple(sorted(set(reasons))),
                    provenance_digest=canonical_digest(provenance),
                )
            )

        groups = tuple(
            EvidenceGroupSummary(
                group_id=group,
                cap=cap_by_group[group],
                requested_weight=requested_by_group[group],
                group_capped_weight=requested_by_group[group] * group_scales[group],
                effective_weight=(
                    requested_by_group[group] * group_scales[group] * task_scale
                ),
            )
            for group in sorted(requested_by_group)
        )
        total_weight = sum(record.effective_weight for record in records)
        certification_weight = sum(
            record.effective_weight
            for record in records
            if record.certification_eligible
        )

        scored = [
            (criterion.score_weight, evaluation_by_criterion[criterion.id].score)
            for criterion in task.criteria
            if criterion.id in evaluation_by_criterion
            and evaluation_by_criterion[criterion.id].status is EvaluationStatus.VALID
            and evaluation_by_criterion[criterion.id].score is not None
        ]
        reported_task_score = (
            None
            if not scored
            else sum(weight * score for weight, score in scored)  # type: ignore[operator]
            / sum(weight for weight, _ in scored)
        )
        evidence_score = (
            None
            if total_weight <= 0.0
            else sum(
                record.effective_weight * record.score
                for record in records
                if record.score is not None
            )
            / total_weight
        )
        return EvidenceBundle(
            task_id=task.id,
            task_version=task.version,
            task_digest=task.digest,
            trace_id=evaluation.trace_id,
            evaluation_id=evaluation.id,
            evaluation_digest=evaluation.digest,
            action_trace_digest=full_trace_digest,
            trace=trace,
            records=tuple(records),
            groups=groups,
            reported_task_score=reported_task_score,
            evidence_score=evidence_score,
            total_evidence_weight=total_weight,
            certification_evidence_weight=certification_weight,
            missing_criterion_ids=tuple(missing),
        )


DEFAULT_REDUCER = DeterministicEvidenceReducer()


def reduce_evidence(
    task: LearningTask,
    evaluation: TaskEvaluation,
    actions: Iterable[LearningAction] = (),
) -> EvidenceBundle:
    """Convenience wrapper around :class:`DeterministicEvidenceReducer`."""

    return DEFAULT_REDUCER.reduce(task, evaluation, actions)


__all__ = [
    "SCHEMA_VERSION",
    "MAX_ACTION_COUNTER",
    "ACTION_PAYLOAD_CONTRACTS",
    "ActionKind",
    "ActionPhase",
    "ActionTraceSummary",
    "CheckProgress",
    "CriterionEvaluation",
    "CriterionScale",
    "DEFAULT_REDUCER",
    "DeterministicEvidenceReducer",
    "EvaluationStatus",
    "EvidenceBundle",
    "EvidenceGroupSummary",
    "EvidenceRecord",
    "LearningAction",
    "LearningTask",
    "RubricCriterion",
    "ScorerContract",
    "ScorerKind",
    "TaskEvaluation",
    "TaskModality",
    "action_trace_digest",
    "canonical_digest",
    "canonical_json",
    "reduce_evidence",
    "summarize_actions",
]
