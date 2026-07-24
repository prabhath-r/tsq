# SPDX-License-Identifier: MPL-2.0

"""Fail-closed scorer authority for productive-performance evaluations.

This module is deliberately an adapter boundary, not an artifact runner and
not a learner-model reducer.  A scoring provider receives only immutable task,
trace, and digest identifiers.  It returns criterion observations that do not
contain authority fields.  The registry supplies authority from separately
configured, release-pinnable bindings and normalizes the observations into a
strict :class:`~tsq.evidence.TaskEvaluation`.

There are two normalization paths:

* :meth:`ScoringProviderRegistry.score` may confer deterministic or verified
  human authority because the provider and its manifest binding were
  registered by the application.
* :func:`normalize_imported_evaluation` always remains shadow-only.  A direct
  import cannot promote itself by claiming to be deterministic or human.

Model, imported, unverified-human, unverified-deterministic, and synthetic
results are shadow observations.  In particular, the deterministic synthetic
provider below is useful for tests and ledger exercises, but it is never
competence evidence.  This module does not execute code, commands, tests,
learner artifacts, or callbacks, and it never updates mastery.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Any, Mapping, Protocol, runtime_checkable

from .evidence import (
    ActionPhase,
    CriterionEvaluation,
    EvaluationStatus,
    ScorerContract,
    ScorerKind,
    TaskEvaluation,
    canonical_digest,
    canonical_json,
)


IMPORTED_EVALUATION_SCHEMA_VERSION = 1
SCORER_AUTHORITY_SCHEMA_VERSION = 1
NORMALIZED_SCORING_RESULT_SCHEMA_VERSION = 1

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_CRITERION_FIELDS = frozenset(
    {
        "criterion_id",
        "status",
        "score",
        "outcome_code",
        "phase",
        "source_action_ids",
        "attestation_digest",
        "misconception_ids",
        "reliability",
    }
)
_IMPORTED_EVALUATION_FIELDS = frozenset({"criteria", "schema_version"})

_DIRECT_IMPORT_AUTHORITY_ID = "authority.direct-import-shadow"
_DIRECT_IMPORT_AUTHORITY_MANIFEST_DIGEST = canonical_digest(
    {
        "authority_id": _DIRECT_IMPORT_AUTHORITY_ID,
        "rule": "direct imports cannot confer deterministic or human authority",
        "schema_version": SCORER_AUTHORITY_SCHEMA_VERSION,
    }
)


class ScoringProtocolError(ValueError):
    """A scorer registration, request, or result violated the closed protocol."""


class ProviderNotFoundError(LookupError):
    """An exact provider ID/version pair was not registered."""


class ProviderExecutionError(RuntimeError):
    """A registered provider failed or returned a non-protocol result."""


class NormalizationMode(StrEnum):
    """How a scorer observation crossed the authority boundary."""

    REGISTERED_PROVIDER = "registered_provider"
    DIRECT_IMPORT = "direct_import"


def _require_id(value: object, label: str) -> str:
    if type(value) is not str or not _ID_PATTERN.fullmatch(value):
        raise ScoringProtocolError(
            f"{label} must match {_ID_PATTERN.pattern!r}; free-form text is not allowed."
        )
    return value


def _require_digest(value: object, label: str) -> str:
    if type(value) is not str or not _DIGEST_PATTERN.fullmatch(value):
        raise ScoringProtocolError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _require_positive_int(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ScoringProtocolError(f"{label} must be a positive integer.")
    return value


def _require_unit_interval(value: object, label: str) -> float:
    try:
        valid = (
            type(value) in {int, float}
            and isfinite(float(value))
            and 0.0 <= float(value) <= 1.0
        )
    except (OverflowError, ValueError):
        valid = False
    if not valid:
        raise ScoringProtocolError(f"{label} must be finite and between 0 and 1.")
    return float(value)


def _require_enum(value: object, enum_type: type[StrEnum], label: str) -> None:
    if not isinstance(value, enum_type):
        raise ScoringProtocolError(f"{label} must be a {enum_type.__name__} value.")


def _sorted_unique_ids(values: object, label: str) -> tuple[str, ...]:
    if type(values) is not tuple:
        raise ScoringProtocolError(f"{label} must be a tuple.")
    result = tuple(_require_id(item, f"{label}[]") for item in values)
    if len(result) != len(set(result)):
        raise ScoringProtocolError(f"{label} must not contain duplicates.")
    return tuple(sorted(result))


def _decode_id_list(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise ScoringProtocolError(f"{label} must be a JSON array.")
    return _sorted_unique_ids(tuple(value), label)


def _decode_manifest_terms(
    value: object,
    *,
    identity_field: str,
    label: str,
) -> tuple[tuple[str, str], ...]:
    if type(value) is not list:
        raise ScoringProtocolError(f"{label} must be a JSON array.")
    entries: list[tuple[str, str]] = []
    for index, item in enumerate(value):
        if type(item) is not dict:
            raise ScoringProtocolError(f"{label}[{index}] must be an object.")
        _exact_fields(
            item,
            frozenset({identity_field, "manifest_digest"}),
            f"{label}[{index}]",
        )
        entries.append(
            (
                _require_id(
                    item[identity_field],
                    f"{label}[{index}].{identity_field}",
                ),
                _require_digest(
                    item["manifest_digest"],
                    f"{label}[{index}].manifest_digest",
                ),
            )
        )
    return _manifest_bindings(tuple(entries), label)


def _exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    actual = set(value)
    if actual == expected:
        return
    details: list[str] = []
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        details.append("missing " + ", ".join(missing))
    if unexpected:
        details.append("unexpected " + ", ".join(unexpected))
    raise ScoringProtocolError(f"{label} has " + "; ".join(details) + ".")


def _enum_from_wire(
    value: object, enum_type: type[StrEnum], label: str
) -> StrEnum:
    if type(value) is not str:
        raise ScoringProtocolError(f"{label} must be a string.")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ScoringProtocolError(f"{label} has unknown value {value!r}.") from exc


def _manifest_bindings(
    value: object, label: str
) -> tuple[tuple[str, str], ...]:
    """Validate a closed, immutable semantic-ID to SHA-256 manifest map."""

    if type(value) is not tuple:
        raise ScoringProtocolError(f"{label} must be a tuple.")
    normalized: list[tuple[str, str]] = []
    for item in value:
        if type(item) is not tuple or len(item) != 2:
            raise ScoringProtocolError(
                f"{label} entries must be (semantic_id, digest) tuples."
            )
        semantic_id = _require_id(item[0], f"{label}[].semantic_id")
        digest = _require_digest(item[1], f"{label}[].digest")
        normalized.append((semantic_id, digest))
    semantic_ids = [semantic_id for semantic_id, _ in normalized]
    if len(semantic_ids) != len(set(semantic_ids)):
        raise ScoringProtocolError(f"{label} semantic IDs must be unique.")
    return tuple(sorted(normalized))


def _reject_constant(raw: str) -> None:
    raise ScoringProtocolError(f"Imported evaluation contains invalid number {raw}.")


def _finite_json_float(raw: str) -> float:
    value = float(raw)
    if not isfinite(value):
        raise ScoringProtocolError("Imported evaluation contains a non-finite number.")
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ScoringProtocolError(
                f"Imported evaluation contains duplicate field {key!r}."
            )
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class ScoringRequest:
    """Content-free envelope identifying exactly what should be scored.

    The request intentionally has no artifact body, source code, prose,
    command, test callback, path, or URL.  Providers import already-produced
    observations; artifact execution belongs in a separately isolated system.
    """

    evaluation_id: str
    trace_id: str
    task_id: str
    task_version: int
    task_digest: str
    action_trace_digest: str
    criterion_ids: tuple[str, ...]
    scorer_contract: ScorerContract | None = None

    def __post_init__(self) -> None:
        _require_id(self.evaluation_id, "ScoringRequest.evaluation_id")
        _require_id(self.trace_id, "ScoringRequest.trace_id")
        _require_id(self.task_id, "ScoringRequest.task_id")
        _require_positive_int(self.task_version, "ScoringRequest.task_version")
        _require_digest(self.task_digest, "ScoringRequest.task_digest")
        _require_digest(
            self.action_trace_digest, "ScoringRequest.action_trace_digest"
        )
        criterion_ids = _sorted_unique_ids(
            self.criterion_ids, "ScoringRequest.criterion_ids"
        )
        if not criterion_ids:
            raise ScoringProtocolError(
                "ScoringRequest.criterion_ids must not be empty."
            )
        object.__setattr__(self, "criterion_ids", criterion_ids)
        if self.scorer_contract is not None:
            if not isinstance(self.scorer_contract, ScorerContract):
                raise ScoringProtocolError(
                    "ScoringRequest.scorer_contract must be a ScorerContract."
                )
            if self.scorer_contract.criterion_ids != criterion_ids:
                raise ScoringProtocolError(
                    "ScoringRequest criteria must exactly match its scorer contract."
                )

    @property
    def scorer_contract_digest(self) -> str | None:
        return (
            None
            if self.scorer_contract is None
            else canonical_digest(self.scorer_contract.terms())
        )

    def terms(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "task_version": self.task_version,
            "task_digest": self.task_digest,
            "action_trace_digest": self.action_trace_digest,
            "criterion_ids": list(self.criterion_ids),
            "scorer_contract": (
                None
                if self.scorer_contract is None
                else self.scorer_contract.terms()
            ),
            "scorer_contract_digest": self.scorer_contract_digest,
        }

    @classmethod
    def from_terms(cls, value: object) -> "ScoringRequest":
        if type(value) is not dict:
            raise ScoringProtocolError("Scoring request must be an object.")
        _exact_fields(
            value,
            frozenset(
                {
                    "evaluation_id",
                    "trace_id",
                    "task_id",
                    "task_version",
                    "task_digest",
                    "action_trace_digest",
                    "criterion_ids",
                    "scorer_contract",
                    "scorer_contract_digest",
                }
            ),
            "Scoring request",
        )
        contract_terms = value["scorer_contract"]
        try:
            contract = (
                None
                if contract_terms is None
                else ScorerContract.from_terms(contract_terms)
            )
        except (TypeError, ValueError) as exc:
            raise ScoringProtocolError(
                f"Scoring request contract is invalid: {exc}"
            ) from exc
        request = cls(
            evaluation_id=value["evaluation_id"],
            trace_id=value["trace_id"],
            task_id=value["task_id"],
            task_version=value["task_version"],
            task_digest=value["task_digest"],
            action_trace_digest=value["action_trace_digest"],
            criterion_ids=_decode_id_list(
                value["criterion_ids"], "Scoring request criterion_ids"
            ),
            scorer_contract=contract,
        )
        if (
            value["scorer_contract_digest"]
            != request.scorer_contract_digest
            or canonical_json(request.terms()) != canonical_json(value)
        ):
            raise ScoringProtocolError(
                "Scoring request terms are not canonical or its contract "
                "digest does not match."
            )
        return request


@dataclass(frozen=True, slots=True)
class ImportedCriterionResult:
    """Authority-free wire observation for one rubric criterion."""

    criterion_id: str
    status: EvaluationStatus
    score: float | None
    outcome_code: str
    phase: ActionPhase
    source_action_ids: tuple[str, ...] = ()
    attestation_digest: str | None = None
    misconception_ids: tuple[str, ...] = ()
    reliability: float = 1.0

    def __post_init__(self) -> None:
        _require_id(
            self.criterion_id, "ImportedCriterionResult.criterion_id"
        )
        _require_enum(
            self.status, EvaluationStatus, "ImportedCriterionResult.status"
        )
        if self.status is EvaluationStatus.VALID:
            if self.score is None:
                raise ScoringProtocolError(
                    "A valid imported criterion result requires a score."
                )
            object.__setattr__(
                self,
                "score",
                _require_unit_interval(
                    self.score, "ImportedCriterionResult.score"
                ),
            )
        elif self.score is not None:
            raise ScoringProtocolError(
                "An invalid or missing imported result cannot carry a score."
            )
        _require_id(self.outcome_code, "ImportedCriterionResult.outcome_code")
        _require_enum(self.phase, ActionPhase, "ImportedCriterionResult.phase")
        object.__setattr__(
            self,
            "source_action_ids",
            _sorted_unique_ids(
                self.source_action_ids,
                "ImportedCriterionResult.source_action_ids",
            ),
        )
        if self.attestation_digest is not None:
            _require_digest(
                self.attestation_digest,
                "ImportedCriterionResult.attestation_digest",
            )
        object.__setattr__(
            self,
            "misconception_ids",
            _sorted_unique_ids(
                self.misconception_ids,
                "ImportedCriterionResult.misconception_ids",
            ),
        )
        if self.score == 1.0 and self.misconception_ids:
            raise ScoringProtocolError(
                "A fully successful result cannot assert an observed misconception."
            )
        object.__setattr__(
            self,
            "reliability",
            _require_unit_interval(
                self.reliability, "ImportedCriterionResult.reliability"
            ),
        )

    @classmethod
    def from_terms(cls, value: object) -> "ImportedCriterionResult":
        if type(value) is not dict:
            raise ScoringProtocolError(
                "Imported criterion result must be a JSON object."
            )
        _exact_fields(value, _CRITERION_FIELDS, "Imported criterion result")
        attestation = value["attestation_digest"]
        if attestation is not None:
            _require_digest(attestation, "attestation_digest")
        return cls(
            criterion_id=_require_id(value["criterion_id"], "criterion_id"),
            status=_enum_from_wire(
                value["status"], EvaluationStatus, "status"
            ),  # type: ignore[arg-type]
            score=value["score"],
            outcome_code=_require_id(value["outcome_code"], "outcome_code"),
            phase=_enum_from_wire(
                value["phase"], ActionPhase, "phase"
            ),  # type: ignore[arg-type]
            source_action_ids=_decode_id_list(
                value["source_action_ids"], "source_action_ids"
            ),
            attestation_digest=attestation,
            misconception_ids=_decode_id_list(
                value["misconception_ids"], "misconception_ids"
            ),
            reliability=value["reliability"],
        )

    def terms(self) -> dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "status": self.status.value,
            "score": self.score,
            "outcome_code": self.outcome_code,
            "phase": self.phase.value,
            "source_action_ids": list(self.source_action_ids),
            "attestation_digest": self.attestation_digest,
            "misconception_ids": list(self.misconception_ids),
            "reliability": self.reliability,
        }


@dataclass(frozen=True, slots=True)
class ImportedEvaluation:
    """Strict, authority-free set of imported criterion observations."""

    criteria: tuple[ImportedCriterionResult, ...]
    schema_version: int = IMPORTED_EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.criteria) is not tuple or any(
            not isinstance(item, ImportedCriterionResult) for item in self.criteria
        ):
            raise ScoringProtocolError(
                "ImportedEvaluation.criteria must be a tuple of "
                "ImportedCriterionResult values."
            )
        criterion_ids = [item.criterion_id for item in self.criteria]
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ScoringProtocolError(
                "ImportedEvaluation criterion IDs must be unique."
            )
        object.__setattr__(
            self,
            "criteria",
            tuple(sorted(self.criteria, key=lambda item: item.criterion_id)),
        )
        if self.schema_version != IMPORTED_EVALUATION_SCHEMA_VERSION:
            raise ScoringProtocolError(
                "ImportedEvaluation.schema_version must be "
                f"{IMPORTED_EVALUATION_SCHEMA_VERSION}."
            )

    @classmethod
    def from_terms(cls, value: object) -> "ImportedEvaluation":
        if type(value) is not dict:
            raise ScoringProtocolError(
                "Imported evaluation must be a JSON object."
            )
        _exact_fields(
            value, _IMPORTED_EVALUATION_FIELDS, "Imported evaluation"
        )
        if value["schema_version"] != IMPORTED_EVALUATION_SCHEMA_VERSION:
            raise ScoringProtocolError(
                "Imported evaluation has an unsupported schema_version."
            )
        raw_criteria = value["criteria"]
        if type(raw_criteria) is not list:
            raise ScoringProtocolError(
                "Imported evaluation criteria must be a JSON array."
            )
        return cls(
            criteria=tuple(
                ImportedCriterionResult.from_terms(item)
                for item in raw_criteria
            )
        )

    @classmethod
    def from_json(cls, raw: str) -> "ImportedEvaluation":
        if type(raw) is not str:
            raise ScoringProtocolError("Imported evaluation JSON must be a string.")
        try:
            value = json.loads(
                raw,
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_constant,
                parse_float=_finite_json_float,
            )
        except ScoringProtocolError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ScoringProtocolError(
                f"Imported evaluation is not valid strict JSON: {exc}"
            ) from exc
        return cls.from_terms(value)

    @property
    def digest(self) -> str:
        return canonical_digest(self.terms())

    def terms(self) -> dict[str, Any]:
        return {
            "criteria": [item.terms() for item in self.criteria],
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class ProviderAuthorityBinding:
    """Application-configured trust binding for one exact provider version.

    ``verified`` is configuration, never provider output.  Deterministic
    authority must be closed over at least one check-set or artifact manifest.
    Model and imported providers cannot be configured as verified.
    """

    provider_id: str
    provider_version: str
    declared_kind: ScorerKind
    authority_id: str
    authority_manifest_digest: str
    check_set_manifests: tuple[tuple[str, str], ...] = ()
    artifact_manifests: tuple[tuple[str, str], ...] = ()
    verified: bool = False
    schema_version: int = SCORER_AUTHORITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_id(self.provider_id, "ProviderAuthorityBinding.provider_id")
        _require_id(
            self.provider_version, "ProviderAuthorityBinding.provider_version"
        )
        _require_enum(
            self.declared_kind,
            ScorerKind,
            "ProviderAuthorityBinding.declared_kind",
        )
        _require_id(self.authority_id, "ProviderAuthorityBinding.authority_id")
        _require_digest(
            self.authority_manifest_digest,
            "ProviderAuthorityBinding.authority_manifest_digest",
        )
        object.__setattr__(
            self,
            "check_set_manifests",
            _manifest_bindings(
                self.check_set_manifests,
                "ProviderAuthorityBinding.check_set_manifests",
            ),
        )
        object.__setattr__(
            self,
            "artifact_manifests",
            _manifest_bindings(
                self.artifact_manifests,
                "ProviderAuthorityBinding.artifact_manifests",
            ),
        )
        if type(self.verified) is not bool:
            raise ScoringProtocolError(
                "ProviderAuthorityBinding.verified must be bool."
            )
        if self.verified and self.declared_kind in {
            ScorerKind.MODEL,
            ScorerKind.IMPORTED,
        }:
            raise ScoringProtocolError(
                "Model and imported providers cannot receive verified authority."
            )
        if (
            self.verified
            and self.declared_kind is ScorerKind.DETERMINISTIC
            and not self.check_set_manifests
            and not self.artifact_manifests
        ):
            raise ScoringProtocolError(
                "Verified deterministic authority requires a closed check-set "
                "or artifact manifest."
            )
        if self.schema_version != SCORER_AUTHORITY_SCHEMA_VERSION:
            raise ScoringProtocolError(
                "ProviderAuthorityBinding.schema_version must be "
                f"{SCORER_AUTHORITY_SCHEMA_VERSION}."
            )

    @property
    def key(self) -> tuple[str, str]:
        return (self.provider_id, self.provider_version)

    @property
    def digest(self) -> str:
        return canonical_digest(self.terms())

    def terms(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "declared_kind": self.declared_kind.value,
            "authority_id": self.authority_id,
            "authority_manifest_digest": self.authority_manifest_digest,
            "check_set_manifests": [
                {"check_set_id": semantic_id, "manifest_digest": digest}
                for semantic_id, digest in self.check_set_manifests
            ],
            "artifact_manifests": [
                {"artifact_kind": semantic_id, "manifest_digest": digest}
                for semantic_id, digest in self.artifact_manifests
            ],
            "verified": self.verified,
            "schema_version": self.schema_version,
        }


@runtime_checkable
class TaskScoringProvider(Protocol):
    """Port for importing already-produced, content-free scorer observations."""

    provider_id: str
    provider_version: str
    declared_kind: ScorerKind
    synthetic: bool

    def score(self, request: ScoringRequest) -> ImportedEvaluation:
        """Return observations without executing learner artifacts."""


@dataclass(frozen=True, slots=True)
class RegisteredProvider:
    """Stable list/inspection view of one registry entry."""

    provider_id: str
    provider_version: str
    declared_kind: ScorerKind
    authority_id: str
    authority_manifest_digest: str
    binding_digest: str
    check_set_manifests: tuple[tuple[str, str], ...]
    artifact_manifests: tuple[tuple[str, str], ...]
    verified: bool
    synthetic: bool

    def __post_init__(self) -> None:
        if type(self.synthetic) is not bool:
            raise ScoringProtocolError(
                "RegisteredProvider.synthetic must be bool."
            )
        try:
            binding = ProviderAuthorityBinding(
                provider_id=self.provider_id,
                provider_version=self.provider_version,
                declared_kind=self.declared_kind,
                authority_id=self.authority_id,
                authority_manifest_digest=self.authority_manifest_digest,
                check_set_manifests=self.check_set_manifests,
                artifact_manifests=self.artifact_manifests,
                verified=self.verified,
            )
        except ScoringProtocolError as exc:
            raise ScoringProtocolError(
                "RegisteredProvider authority binding is invalid."
            ) from exc
        _require_digest(
            self.binding_digest, "RegisteredProvider.binding_digest"
        )
        if self.binding_digest != binding.digest:
            raise ScoringProtocolError(
                "RegisteredProvider binding digest does not match."
            )
        object.__setattr__(
            self, "check_set_manifests", binding.check_set_manifests
        )
        object.__setattr__(
            self, "artifact_manifests", binding.artifact_manifests
        )
        if self.synthetic and not self.provider_id.startswith("synthetic."):
            raise ScoringProtocolError(
                "Synthetic provider IDs must start with 'synthetic.'."
            )
        if self.provider_id.startswith("synthetic.") and not self.synthetic:
            raise ScoringProtocolError(
                "Providers in the synthetic namespace must be marked synthetic."
            )
        if self.synthetic and self.verified:
            raise ScoringProtocolError(
                "Synthetic providers cannot receive verified authority."
            )

    @property
    def key(self) -> tuple[str, str]:
        return (self.provider_id, self.provider_version)

    @property
    def shadow_only(self) -> bool:
        return (
            self.synthetic
            or not self.verified
            or self.declared_kind in {ScorerKind.MODEL, ScorerKind.IMPORTED}
        )

    def terms(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "declared_kind": self.declared_kind.value,
            "authority_id": self.authority_id,
            "authority_manifest_digest": self.authority_manifest_digest,
            "binding_digest": self.binding_digest,
            "check_set_manifests": [
                {"check_set_id": semantic_id, "manifest_digest": digest}
                for semantic_id, digest in self.check_set_manifests
            ],
            "artifact_manifests": [
                {"artifact_kind": semantic_id, "manifest_digest": digest}
                for semantic_id, digest in self.artifact_manifests
            ],
            "verified": self.verified,
            "synthetic": self.synthetic,
            "shadow_only": self.shadow_only,
        }

    @classmethod
    def from_terms(cls, value: object) -> "RegisteredProvider":
        if type(value) is not dict:
            raise ScoringProtocolError("Registered provider must be an object.")
        _exact_fields(
            value,
            frozenset(
                {
                    "provider_id",
                    "provider_version",
                    "declared_kind",
                    "authority_id",
                    "authority_manifest_digest",
                    "binding_digest",
                    "check_set_manifests",
                    "artifact_manifests",
                    "verified",
                    "synthetic",
                    "shadow_only",
                }
            ),
            "Registered provider",
        )
        declared_kind = _enum_from_wire(
            value["declared_kind"],
            ScorerKind,
            "Registered provider declared_kind",
        )
        if type(value["verified"]) is not bool:
            raise ScoringProtocolError(
                "Registered provider verified must be bool."
            )
        if type(value["synthetic"]) is not bool:
            raise ScoringProtocolError(
                "Registered provider synthetic must be bool."
            )
        binding = ProviderAuthorityBinding(
            provider_id=value["provider_id"],
            provider_version=value["provider_version"],
            declared_kind=declared_kind,  # type: ignore[arg-type]
            authority_id=value["authority_id"],
            authority_manifest_digest=value["authority_manifest_digest"],
            check_set_manifests=_decode_manifest_terms(
                value["check_set_manifests"],
                identity_field="check_set_id",
                label="Registered provider check_set_manifests",
            ),
            artifact_manifests=_decode_manifest_terms(
                value["artifact_manifests"],
                identity_field="artifact_kind",
                label="Registered provider artifact_manifests",
            ),
            verified=value["verified"],
        )
        provider = cls(
            provider_id=binding.provider_id,
            provider_version=binding.provider_version,
            declared_kind=binding.declared_kind,
            authority_id=binding.authority_id,
            authority_manifest_digest=binding.authority_manifest_digest,
            binding_digest=value["binding_digest"],
            check_set_manifests=binding.check_set_manifests,
            artifact_manifests=binding.artifact_manifests,
            verified=binding.verified,
            synthetic=value["synthetic"],
        )
        if (
            provider.binding_digest != binding.digest
            or canonical_json(provider.terms()) != canonical_json(value)
        ):
            raise ScoringProtocolError(
                "Registered provider terms are not canonical or its binding "
                "digest does not match."
            )
        return provider


@dataclass(frozen=True, slots=True)
class CriterionAuthorityDecision:
    """Auditable effective authority assigned to one normalized result."""

    criterion_id: str
    declared_kind: ScorerKind
    effective_kind: ScorerKind
    reason_code: str

    def __post_init__(self) -> None:
        _require_id(
            self.criterion_id, "CriterionAuthorityDecision.criterion_id"
        )
        _require_enum(
            self.declared_kind,
            ScorerKind,
            "CriterionAuthorityDecision.declared_kind",
        )
        _require_enum(
            self.effective_kind,
            ScorerKind,
            "CriterionAuthorityDecision.effective_kind",
        )
        _require_id(
            self.reason_code, "CriterionAuthorityDecision.reason_code"
        )

    @property
    def shadow_only(self) -> bool:
        return self.effective_kind in {
            ScorerKind.MODEL,
            ScorerKind.IMPORTED,
        }

    def terms(self) -> dict[str, str | bool]:
        return {
            "criterion_id": self.criterion_id,
            "declared_kind": self.declared_kind.value,
            "effective_kind": self.effective_kind.value,
            "reason_code": self.reason_code,
            "shadow_only": self.shadow_only,
        }

    @classmethod
    def from_terms(cls, value: object) -> "CriterionAuthorityDecision":
        if type(value) is not dict:
            raise ScoringProtocolError("Authority decision must be an object.")
        _exact_fields(
            value,
            frozenset(
                {
                    "criterion_id",
                    "declared_kind",
                    "effective_kind",
                    "reason_code",
                    "shadow_only",
                }
            ),
            "Authority decision",
        )
        decision = cls(
            criterion_id=value["criterion_id"],
            declared_kind=_enum_from_wire(
                value["declared_kind"],
                ScorerKind,
                "Authority decision declared_kind",
            ),  # type: ignore[arg-type]
            effective_kind=_enum_from_wire(
                value["effective_kind"],
                ScorerKind,
                "Authority decision effective_kind",
            ),  # type: ignore[arg-type]
            reason_code=value["reason_code"],
        )
        if canonical_json(decision.terms()) != canonical_json(value):
            raise ScoringProtocolError(
                "Authority decision terms are not canonical."
            )
        return decision


@dataclass(frozen=True, slots=True)
class NormalizedScoringResult:
    """Strict evidence evaluation plus its separately auditable authority decision."""

    evaluation: TaskEvaluation
    request: ScoringRequest
    provider: RegisteredProvider
    decisions: tuple[CriterionAuthorityDecision, ...]
    normalization_mode: NormalizationMode
    schema_version: int = NORMALIZED_SCORING_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation, TaskEvaluation):
            raise ScoringProtocolError(
                "NormalizedScoringResult.evaluation must be a TaskEvaluation."
            )
        if not isinstance(self.request, ScoringRequest):
            raise ScoringProtocolError(
                "NormalizedScoringResult.request must be a ScoringRequest."
            )
        if not isinstance(self.provider, RegisteredProvider):
            raise ScoringProtocolError(
                "NormalizedScoringResult.provider must be a RegisteredProvider."
            )
        if type(self.decisions) is not tuple or any(
            not isinstance(item, CriterionAuthorityDecision)
            for item in self.decisions
        ):
            raise ScoringProtocolError(
                "NormalizedScoringResult.decisions must be authority decisions."
            )
        _require_enum(
            self.normalization_mode,
            NormalizationMode,
            "NormalizedScoringResult.normalization_mode",
        )
        if self.schema_version != NORMALIZED_SCORING_RESULT_SCHEMA_VERSION:
            raise ScoringProtocolError(
                "NormalizedScoringResult.schema_version must be "
                f"{NORMALIZED_SCORING_RESULT_SCHEMA_VERSION}."
            )
        expected_envelope = (
            self.request.evaluation_id,
            self.request.trace_id,
            self.request.task_id,
            self.request.task_version,
            self.request.task_digest,
            self.request.action_trace_digest,
        )
        actual_envelope = (
            self.evaluation.id,
            self.evaluation.trace_id,
            self.evaluation.task_id,
            self.evaluation.task_version,
            self.evaluation.task_digest,
            self.evaluation.action_trace_digest,
        )
        if actual_envelope != expected_envelope:
            raise ScoringProtocolError(
                "NormalizedScoringResult evaluation does not match its request."
            )
        try:
            provider_binding = ProviderAuthorityBinding(
                provider_id=self.provider.provider_id,
                provider_version=self.provider.provider_version,
                declared_kind=self.provider.declared_kind,
                authority_id=self.provider.authority_id,
                authority_manifest_digest=(
                    self.provider.authority_manifest_digest
                ),
                check_set_manifests=self.provider.check_set_manifests,
                artifact_manifests=self.provider.artifact_manifests,
                verified=self.provider.verified,
            )
        except ScoringProtocolError as exc:
            raise ScoringProtocolError(
                "NormalizedScoringResult provider binding is invalid."
            ) from exc
        if provider_binding.digest != self.provider.binding_digest:
            raise ScoringProtocolError(
                "NormalizedScoringResult provider binding digest does not match."
            )
        contract = self.request.scorer_contract
        if contract is not None:
            if (
                contract.key
                != (
                    self.provider.declared_kind,
                    self.provider.provider_id,
                    self.provider.provider_version,
                )
                or contract.authority_id != self.provider.authority_id
                or contract.authority_manifest_digest
                != self.provider.authority_manifest_digest
                or contract.check_set_manifests
                != self.provider.check_set_manifests
                or contract.artifact_manifests
                != self.provider.artifact_manifests
            ):
                raise ScoringProtocolError(
                    "NormalizedScoringResult request contract does not match "
                    "its provider."
                )
        if self.normalization_mode is NormalizationMode.DIRECT_IMPORT:
            if (
                contract is not None
                or self.provider.verified
                or self.provider.synthetic
                or self.provider.authority_id != _DIRECT_IMPORT_AUTHORITY_ID
                or self.provider.authority_manifest_digest
                != _DIRECT_IMPORT_AUTHORITY_MANIFEST_DIGEST
            ):
                raise ScoringProtocolError(
                    "NormalizedScoringResult direct-import authority is invalid."
                )
        elif (
            self.provider.verified
            and not self.provider.shadow_only
            and contract is None
        ):
            raise ScoringProtocolError(
                "NormalizedScoringResult verified provider lacks its request "
                "contract."
            )
        evaluation_by_id = {
            item.criterion_id: item for item in self.evaluation.criteria
        }
        if set(evaluation_by_id) != set(self.request.criterion_ids):
            raise ScoringProtocolError(
                "NormalizedScoringResult evaluation criteria must exactly "
                "match its request."
            )
        decisions_by_id = {item.criterion_id: item for item in self.decisions}
        if len(decisions_by_id) != len(self.decisions):
            raise ScoringProtocolError(
                "NormalizedScoringResult decision criterion IDs must be unique."
            )
        if set(evaluation_by_id) != set(decisions_by_id):
            raise ScoringProtocolError(
                "NormalizedScoringResult decisions must exactly cover its criteria."
            )
        for criterion_id, decision in decisions_by_id.items():
            criterion = evaluation_by_id[criterion_id]
            expected_kind, expected_reason = _authority_decision(
                self.provider.declared_kind,
                verified=self.provider.verified,
                synthetic=self.provider.synthetic,
                direct_import=(
                    self.normalization_mode is NormalizationMode.DIRECT_IMPORT
                ),
                attestation_digest=criterion.attestation_digest,
            )
            if (
                criterion.scorer_kind is not decision.effective_kind
                or criterion.scorer_id != self.provider.provider_id
                or criterion.scorer_version != self.provider.provider_version
                or decision.declared_kind is not self.provider.declared_kind
                or decision.effective_kind is not expected_kind
                or decision.reason_code != expected_reason
            ):
                raise ScoringProtocolError(
                    f"Criterion {criterion_id} authority does not match its "
                    "evaluation and provider."
                )
        object.__setattr__(
            self,
            "decisions",
            tuple(sorted(self.decisions, key=lambda item: item.criterion_id)),
        )

    @property
    def shadow_only(self) -> bool:
        return all(item.shadow_only for item in self.decisions)

    @property
    def digest(self) -> str:
        return canonical_digest(self.terms())

    def terms(self) -> dict[str, Any]:
        return {
            "evaluation": self.evaluation.terms(),
            "request": self.request.terms(),
            "provider": self.provider.terms(),
            "decisions": [item.terms() for item in self.decisions],
            "normalization_mode": self.normalization_mode.value,
            "shadow_only": self.shadow_only,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_terms(cls, value: object) -> "NormalizedScoringResult":
        if type(value) is not dict:
            raise ScoringProtocolError(
                "Normalized scoring result must be an object."
            )
        _exact_fields(
            value,
            frozenset(
                {
                    "evaluation",
                    "request",
                    "provider",
                    "decisions",
                    "normalization_mode",
                    "shadow_only",
                    "schema_version",
                }
            ),
            "Normalized scoring result",
        )
        try:
            evaluation = TaskEvaluation.from_terms(value["evaluation"])
        except (TypeError, ValueError) as exc:
            raise ScoringProtocolError(
                f"Normalized scoring result evaluation is invalid: {exc}"
            ) from exc
        raw_decisions = value["decisions"]
        if type(raw_decisions) is not list:
            raise ScoringProtocolError(
                "Normalized scoring result decisions must be an array."
            )
        result = cls(
            evaluation=evaluation,
            request=ScoringRequest.from_terms(value["request"]),
            provider=RegisteredProvider.from_terms(value["provider"]),
            decisions=tuple(
                CriterionAuthorityDecision.from_terms(item)
                for item in raw_decisions
            ),
            normalization_mode=_enum_from_wire(
                value["normalization_mode"],
                NormalizationMode,
                "Normalized scoring result normalization_mode",
            ),  # type: ignore[arg-type]
            schema_version=value["schema_version"],
        )
        if canonical_json(result.terms()) != canonical_json(value):
            raise ScoringProtocolError(
                "Normalized scoring result terms are not canonical."
            )
        return result


@dataclass(frozen=True, slots=True)
class _ProviderRegistration:
    provider: TaskScoringProvider
    binding: ProviderAuthorityBinding
    summary: RegisteredProvider


def _authority_decision(
    declared_kind: ScorerKind,
    *,
    verified: bool,
    synthetic: bool,
    direct_import: bool,
    attestation_digest: str | None,
) -> tuple[ScorerKind, str]:
    if synthetic:
        return (ScorerKind.IMPORTED, "synthetic_provider_shadow_only")
    if direct_import:
        if declared_kind is ScorerKind.MODEL:
            return (ScorerKind.MODEL, "model_score_shadow_only")
        if declared_kind is ScorerKind.HUMAN:
            return (ScorerKind.IMPORTED, "unverified_human_shadow_only")
        if declared_kind is ScorerKind.DETERMINISTIC:
            return (
                ScorerKind.IMPORTED,
                "direct_import_cannot_claim_authority",
            )
        return (ScorerKind.IMPORTED, "imported_score_unadjudicated")
    if declared_kind is ScorerKind.MODEL:
        return (ScorerKind.MODEL, "model_score_shadow_only")
    if declared_kind is ScorerKind.IMPORTED:
        return (ScorerKind.IMPORTED, "imported_score_unadjudicated")
    if not verified:
        reason = (
            "unverified_human_shadow_only"
            if declared_kind is ScorerKind.HUMAN
            else "unverified_deterministic_shadow_only"
        )
        return (ScorerKind.IMPORTED, reason)
    if declared_kind is ScorerKind.HUMAN:
        if attestation_digest is None:
            return (
                ScorerKind.IMPORTED,
                "missing_verified_human_attestation",
            )
        return (ScorerKind.HUMAN, "verified_human_authority")
    return (ScorerKind.DETERMINISTIC, "verified_deterministic_authority")


def _normalize(
    request: ScoringRequest,
    imported: ImportedEvaluation,
    provider: RegisteredProvider,
    *,
    mode: NormalizationMode,
) -> NormalizedScoringResult:
    if not isinstance(request, ScoringRequest):
        raise ScoringProtocolError("request must be a ScoringRequest.")
    if not isinstance(imported, ImportedEvaluation):
        raise ScoringProtocolError("imported must be an ImportedEvaluation.")
    actual_ids = tuple(item.criterion_id for item in imported.criteria)
    if actual_ids != request.criterion_ids:
        missing = sorted(set(request.criterion_ids) - set(actual_ids))
        unexpected = sorted(set(actual_ids) - set(request.criterion_ids))
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ScoringProtocolError(
            "Imported evaluation must explicitly cover every requested criterion"
            + (": " + "; ".join(details) if details else "")
            + "."
        )

    direct_import = mode is NormalizationMode.DIRECT_IMPORT
    evaluations: list[CriterionEvaluation] = []
    decisions: list[CriterionAuthorityDecision] = []
    for item in imported.criteria:
        effective_kind, reason_code = _authority_decision(
            provider.declared_kind,
            verified=provider.verified,
            synthetic=provider.synthetic,
            direct_import=direct_import,
            attestation_digest=item.attestation_digest,
        )
        evaluations.append(
            CriterionEvaluation(
                criterion_id=item.criterion_id,
                status=item.status,
                score=item.score,
                outcome_code=item.outcome_code,
                phase=item.phase,
                scorer_kind=effective_kind,
                scorer_id=provider.provider_id,
                scorer_version=provider.provider_version,
                source_action_ids=item.source_action_ids,
                attestation_digest=item.attestation_digest,
                misconception_ids=item.misconception_ids,
                reliability=item.reliability,
            )
        )
        decisions.append(
            CriterionAuthorityDecision(
                criterion_id=item.criterion_id,
                declared_kind=provider.declared_kind,
                effective_kind=effective_kind,
                reason_code=reason_code,
            )
        )
    evaluation = TaskEvaluation(
        id=request.evaluation_id,
        trace_id=request.trace_id,
        task_id=request.task_id,
        task_version=request.task_version,
        task_digest=request.task_digest,
        action_trace_digest=request.action_trace_digest,
        criteria=tuple(evaluations),
    )
    return NormalizedScoringResult(
        evaluation=evaluation,
        request=request,
        provider=provider,
        decisions=tuple(decisions),
        normalization_mode=mode,
    )


def normalize_imported_evaluation(
    request: ScoringRequest,
    imported: ImportedEvaluation,
    *,
    provider_id: str,
    provider_version: str,
    declared_kind: ScorerKind = ScorerKind.IMPORTED,
) -> NormalizedScoringResult:
    """Normalize direct input without allowing it to acquire trusted authority.

    ``declared_kind`` is retained solely as provenance.  A direct
    ``DETERMINISTIC`` or ``HUMAN`` declaration is normalized to ``IMPORTED``;
    a model remains ``MODEL`` and therefore shadow-only under the evidence
    reducer.
    """

    _require_id(provider_id, "provider_id")
    _require_id(provider_version, "provider_version")
    _require_enum(declared_kind, ScorerKind, "declared_kind")
    binding = ProviderAuthorityBinding(
        provider_id=provider_id,
        provider_version=provider_version,
        declared_kind=declared_kind,
        authority_id=_DIRECT_IMPORT_AUTHORITY_ID,
        authority_manifest_digest=_DIRECT_IMPORT_AUTHORITY_MANIFEST_DIGEST,
        verified=False,
    )
    provider = RegisteredProvider(
        provider_id=provider_id,
        provider_version=provider_version,
        declared_kind=declared_kind,
        authority_id=binding.authority_id,
        authority_manifest_digest=binding.authority_manifest_digest,
        binding_digest=binding.digest,
        check_set_manifests=(),
        artifact_manifests=(),
        verified=False,
        synthetic=False,
    )
    return _normalize(
        request,
        imported,
        provider,
        mode=NormalizationMode.DIRECT_IMPORT,
    )


class ScoringProviderRegistry:
    """Exact-version, no-overwrite registry for scorer adapters and authority."""

    def __init__(self, *, allow_synthetic: bool = False) -> None:
        if type(allow_synthetic) is not bool:
            raise ScoringProtocolError("allow_synthetic must be bool.")
        self._allow_synthetic = allow_synthetic
        self._providers: dict[tuple[str, str], _ProviderRegistration] = {}

    def register(
        self,
        provider: TaskScoringProvider,
        binding: ProviderAuthorityBinding,
    ) -> RegisteredProvider:
        """Register one exact provider version; an existing key is immutable."""

        if not isinstance(binding, ProviderAuthorityBinding):
            raise ScoringProtocolError(
                "binding must be a ProviderAuthorityBinding."
            )
        try:
            provider_id = provider.provider_id
            provider_version = provider.provider_version
            declared_kind = provider.declared_kind
            synthetic = provider.synthetic
            score_method = provider.score
        except (AttributeError, TypeError) as exc:
            raise ScoringProtocolError(
                "provider does not implement TaskScoringProvider."
            ) from exc
        _require_id(provider_id, "provider.provider_id")
        _require_id(provider_version, "provider.provider_version")
        _require_enum(
            declared_kind, ScorerKind, "provider.declared_kind"
        )
        if type(synthetic) is not bool:
            raise ScoringProtocolError("provider.synthetic must be bool.")
        if not callable(score_method):
            raise ScoringProtocolError("provider.score must be callable.")
        if binding.key != (provider_id, provider_version):
            raise ScoringProtocolError(
                "Provider identity does not match its authority binding."
            )
        if binding.declared_kind is not declared_kind:
            raise ScoringProtocolError(
                "Provider kind does not match its authority binding."
            )
        if synthetic and not provider_id.startswith("synthetic."):
            raise ScoringProtocolError(
                "Synthetic provider IDs must start with 'synthetic.'."
            )
        if provider_id.startswith("synthetic.") and not synthetic:
            raise ScoringProtocolError(
                "Providers in the synthetic namespace must be marked synthetic."
            )
        if synthetic and not self._allow_synthetic:
            raise ScoringProtocolError(
                "Synthetic providers require allow_synthetic=True."
            )
        if synthetic and binding.verified:
            raise ScoringProtocolError(
                "Synthetic providers cannot receive verified authority."
            )
        key = (provider_id, provider_version)
        if key in self._providers:
            raise ScoringProtocolError(
                f"Provider {provider_id}@{provider_version} is already registered."
            )
        summary = RegisteredProvider(
            provider_id=provider_id,
            provider_version=provider_version,
            declared_kind=declared_kind,
            authority_id=binding.authority_id,
            authority_manifest_digest=binding.authority_manifest_digest,
            binding_digest=binding.digest,
            check_set_manifests=binding.check_set_manifests,
            artifact_manifests=binding.artifact_manifests,
            verified=binding.verified,
            synthetic=synthetic,
        )
        self._providers[key] = _ProviderRegistration(
            provider=provider,
            binding=binding,
            summary=summary,
        )
        return summary

    def list(self) -> tuple[RegisteredProvider, ...]:
        """Return stable, sorted, content-free provider inspection records."""

        return tuple(
            self._providers[key].summary for key in sorted(self._providers)
        )

    def inspect(
        self, provider_id: str, provider_version: str
    ) -> RegisteredProvider:
        """Inspect one exact provider version without invoking it."""

        _require_id(provider_id, "provider_id")
        _require_id(provider_version, "provider_version")
        key = (provider_id, provider_version)
        try:
            return self._providers[key].summary
        except KeyError as exc:
            raise ProviderNotFoundError(
                f"Provider {provider_id}@{provider_version} is not registered."
            ) from exc

    def score(
        self,
        provider_id: str,
        provider_version: str,
        request: ScoringRequest,
    ) -> NormalizedScoringResult:
        """Invoke a registered importer and return a strict TaskEvaluation."""

        summary = self.inspect(provider_id, provider_version)
        registration = self._providers[(provider_id, provider_version)]
        provider = registration.provider
        # Provider identity is rechecked to catch mutable or swapped adapters.
        try:
            current_identity = (
                provider.provider_id,
                provider.provider_version,
                provider.declared_kind,
                provider.synthetic,
            )
        except AttributeError as exc:
            raise ProviderExecutionError(
                "Registered provider identity became unavailable."
            ) from exc
        expected_identity = (
            summary.provider_id,
            summary.provider_version,
            summary.declared_kind,
            summary.synthetic,
        )
        if current_identity != expected_identity:
            raise ProviderExecutionError(
                "Registered provider identity changed after registration."
            )
        if not isinstance(request, ScoringRequest):
            raise ScoringProtocolError("request must be a ScoringRequest.")
        self._validate_task_contract(summary, request)
        try:
            imported = provider.score(request)
        except Exception as exc:
            raise ProviderExecutionError(
                f"Provider {provider_id}@{provider_version} failed."
            ) from exc
        if not isinstance(imported, ImportedEvaluation):
            raise ProviderExecutionError(
                f"Provider {provider_id}@{provider_version} returned "
                "a non-protocol result."
            )
        try:
            final_identity = (
                provider.provider_id,
                provider.provider_version,
                provider.declared_kind,
                provider.synthetic,
            )
        except (AttributeError, TypeError) as exc:
            raise ProviderExecutionError(
                "Registered provider identity became unavailable after scoring."
            ) from exc
        if final_identity != expected_identity:
            raise ProviderExecutionError(
                "Registered provider identity changed while scoring."
            )
        return _normalize(
            request,
            imported,
            summary,
            mode=NormalizationMode.REGISTERED_PROVIDER,
        )

    @staticmethod
    def _validate_task_contract(
        provider: RegisteredProvider, request: ScoringRequest
    ) -> None:
        """Bind evidence-capable scoring to the release-pinned task contract."""

        contract = request.scorer_contract
        if contract is None:
            if provider.verified and not provider.shadow_only:
                raise ScoringProtocolError(
                    "Verified scoring requires the release-pinned scorer contract."
                )
            return
        expected_key = (
            provider.declared_kind,
            provider.provider_id,
            provider.provider_version,
        )
        if contract.key != expected_key:
            raise ScoringProtocolError(
                "Scoring request contract does not match the registered provider."
            )
        expected_authority = (
            provider.authority_id,
            provider.authority_manifest_digest,
            provider.check_set_manifests,
            provider.artifact_manifests,
        )
        actual_authority = (
            contract.authority_id,
            contract.authority_manifest_digest,
            contract.check_set_manifests,
            contract.artifact_manifests,
        )
        if actual_authority != expected_authority:
            raise ScoringProtocolError(
                "Scoring request contract does not match the provider manifests."
            )


class SyntheticDeterministicProvider:
    """Deterministic, test-only scorer importer that is always shadow-only.

    It returns one immutable fixture and performs no I/O or artifact execution.
    The registry requires explicit synthetic opt-in and refuses verified
    authority for every provider carrying the synthetic marker.
    """

    declared_kind = ScorerKind.DETERMINISTIC
    synthetic = True

    def __init__(
        self,
        evaluation: ImportedEvaluation,
        *,
        provider_id: str = "synthetic.fixed-performance-scorer",
        provider_version: str = "test-v1",
        check_set_manifests: tuple[tuple[str, str], ...] = (),
        artifact_manifests: tuple[tuple[str, str], ...] = (),
    ) -> None:
        if not isinstance(evaluation, ImportedEvaluation):
            raise ScoringProtocolError(
                "Synthetic provider evaluation must be an ImportedEvaluation."
            )
        _require_id(provider_id, "SyntheticDeterministicProvider.provider_id")
        if not provider_id.startswith("synthetic."):
            raise ScoringProtocolError(
                "Synthetic provider IDs must start with 'synthetic.'."
            )
        _require_id(
            provider_version,
            "SyntheticDeterministicProvider.provider_version",
        )
        self.provider_id = provider_id
        self.provider_version = provider_version
        self._evaluation = evaluation
        self._check_set_manifests = _manifest_bindings(
            check_set_manifests,
            "SyntheticDeterministicProvider.check_set_manifests",
        )
        self._artifact_manifests = _manifest_bindings(
            artifact_manifests,
            "SyntheticDeterministicProvider.artifact_manifests",
        )

    @property
    def authority_binding(self) -> ProviderAuthorityBinding:
        authority_id = "authority." + self.provider_id
        manifest_digest = canonical_digest(
            {
                "authority_id": authority_id,
                "provider_id": self.provider_id,
                "provider_version": self.provider_version,
                "fixture_digest": self._evaluation.digest,
                "synthetic": True,
            }
        )
        return ProviderAuthorityBinding(
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            declared_kind=self.declared_kind,
            authority_id=authority_id,
            authority_manifest_digest=manifest_digest,
            check_set_manifests=self._check_set_manifests,
            artifact_manifests=self._artifact_manifests,
            verified=False,
        )

    def score(self, request: ScoringRequest) -> ImportedEvaluation:
        if not isinstance(request, ScoringRequest):
            raise ScoringProtocolError("request must be a ScoringRequest.")
        actual = tuple(item.criterion_id for item in self._evaluation.criteria)
        if actual != request.criterion_ids:
            raise ScoringProtocolError(
                "Synthetic fixture does not exactly cover the request criteria."
            )
        return self._evaluation


__all__ = [
    "IMPORTED_EVALUATION_SCHEMA_VERSION",
    "SCORER_AUTHORITY_SCHEMA_VERSION",
    "CriterionAuthorityDecision",
    "ImportedCriterionResult",
    "ImportedEvaluation",
    "NormalizationMode",
    "NormalizedScoringResult",
    "NORMALIZED_SCORING_RESULT_SCHEMA_VERSION",
    "ProviderAuthorityBinding",
    "ProviderExecutionError",
    "ProviderNotFoundError",
    "RegisteredProvider",
    "ScoringProtocolError",
    "ScoringProviderRegistry",
    "ScoringRequest",
    "SyntheticDeterministicProvider",
    "TaskScoringProvider",
    "normalize_imported_evaluation",
]
