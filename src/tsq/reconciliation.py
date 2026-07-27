# SPDX-License-Identifier: MPL-2.0

"""Closed, observational reconciliation for admitted scoring callbacks.

This module does not score artifacts, retry scoring callbacks, update learner
projections, or confer scorer authority.  It defines a content-free request and
receipt protocol for asking a separately registered adapter what happened to an
already-admitted provider operation.

Registration is the trust boundary for the observational adapter.  A receipt's
attestation digest is an immutable commitment supplied by that adapter; this
module makes no claim that the digest is a cryptographic signature or that TSQ
has independently verified one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from math import isfinite
from typing import Any, Mapping, Protocol, runtime_checkable

from .evidence import canonical_digest, canonical_json
from .performance import ImportedEvaluation


RECONCILIATION_SCHEMA_VERSION = 1
RECONCILIATION_AUTHORITY_SCHEMA_VERSION = 1
RECONCILIATION_RESULT_SCHEMA_VERSION = 1

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_REQUEST_FIELDS = frozenset(
    {
        "claim_id",
        "attempt_id",
        "evaluation_id",
        "through_sequence",
        "provider_id",
        "provider_version",
        "action_trace_digest",
        "command_hash",
        "scoring_request_digest",
        "provider_binding_digest",
        "provider_operation_digest",
        "schema_version",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "claim_id",
        "attempt_id",
        "evaluation_id",
        "through_sequence",
        "provider_id",
        "provider_version",
        "reconciler_id",
        "reconciler_version",
        "action_trace_digest",
        "command_hash",
        "scoring_request_digest",
        "provider_binding_digest",
        "outcome",
        "observed_at",
        "completed_at",
        "result_digest",
        "reason_code",
        "provider_operation_digest",
        "provider_receipt_digest",
        "attestation_digest",
        "schema_version",
    }
)
_AUTHORITY_FIELDS = frozenset(
    {
        "provider_id",
        "provider_version",
        "reconciler_id",
        "reconciler_version",
        "manifest_digest",
        "synthetic",
        "can_prove_absence",
        "schema_version",
    }
)
_REGISTERED_FIELDS = frozenset(
    {
        "provider_id",
        "provider_version",
        "reconciler_id",
        "reconciler_version",
        "manifest_digest",
        "binding_digest",
        "synthetic",
        "can_prove_absence",
        "observational_only",
        "skill_authority",
    }
)
_OBSERVATION_FIELDS = frozenset({"receipt", "imported_evaluation"})
_RESULT_FIELDS = frozenset(
    {
        "request",
        "observation",
        "reconciler",
        "schema_version",
        "observational_only",
        "automatic_retry_allowed",
        "projection_applied",
        "certification_applied",
        "skill_authority",
        "cryptographic_verification_claim",
        "attestation_semantics",
    }
)


class ReconciliationProtocolError(ValueError):
    """A reconciliation request, receipt, or authority term was invalid."""


class ReconcilerNotFoundError(LookupError):
    """An exact provider/reconciler identity was not registered."""


class ReconcilerExecutionError(RuntimeError):
    """A trusted observational adapter failed or returned invalid output."""


class ReconciliationOutcome(StrEnum):
    """Closed observations about one previously admitted callback."""

    UNKNOWN = "unknown"
    DEFINITELY_ABSENT = "definitely_absent"
    COMPLETED = "completed"


def _require_id(value: object, label: str) -> str:
    if type(value) is not str or not _ID_PATTERN.fullmatch(value):
        raise ReconciliationProtocolError(
            f"{label} must match {_ID_PATTERN.pattern!r}; "
            "free-form text is not allowed."
        )
    return value


def _require_digest(value: object, label: str) -> str:
    if type(value) is not str or not _DIGEST_PATTERN.fullmatch(value):
        raise ReconciliationProtocolError(
            f"{label} must be a lowercase SHA-256 digest."
        )
    return value


def _require_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ReconciliationProtocolError(
            f"{label} must be a non-negative integer."
        )
    return value


def _require_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ReconciliationProtocolError(f"{label} must be bool.")
    return value


def _require_exact_fields(
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
    raise ReconciliationProtocolError(f"{label} has " + "; ".join(details) + ".")


def _enum_from_wire(
    value: object, enum_type: type[StrEnum], label: str
) -> StrEnum:
    if type(value) is not str:
        raise ReconciliationProtocolError(f"{label} must be a string.")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ReconciliationProtocolError(
            f"{label} has unknown value {value!r}."
        ) from exc


def _canonical_timestamp(value: object, label: str) -> str:
    if type(value) is not str:
        raise ReconciliationProtocolError(
            f"{label} must be an ISO-8601 timestamp string."
        )
    try:
        parsed = datetime.fromisoformat(value)
    except (ValueError, OverflowError) as exc:
        raise ReconciliationProtocolError(
            f"{label} is not a valid timestamp."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ReconciliationProtocolError(
            f"{label} must include a timezone offset."
        )
    canonical = parsed.astimezone(timezone.utc).isoformat()
    if value != canonical:
        raise ReconciliationProtocolError(
            f"{label} must use its canonical UTC ISO-8601 representation."
        )
    return value


def _timestamp_value(value: str) -> datetime:
    """Decode an already validated canonical timestamp."""

    return datetime.fromisoformat(value)


def _reject_constant(raw: str) -> None:
    raise ReconciliationProtocolError(
        f"Reconciliation JSON contains invalid number {raw}."
    )


def _finite_json_float(raw: str) -> float:
    value = float(raw)
    if not isfinite(value):
        raise ReconciliationProtocolError(
            "Reconciliation JSON contains a non-finite number."
        )
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReconciliationProtocolError(
                f"Reconciliation JSON contains duplicate field {key!r}."
            )
        result[key] = value
    return result


def _strict_json_object(raw: str, label: str) -> dict[str, Any]:
    if type(raw) is not str:
        raise ReconciliationProtocolError(f"{label} JSON must be a string.")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
            parse_float=_finite_json_float,
        )
    except ReconciliationProtocolError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ReconciliationProtocolError(
            f"{label} is not valid strict JSON: {exc}"
        ) from exc
    if type(value) is not dict:
        raise ReconciliationProtocolError(f"{label} must be a JSON object.")
    return value


def provider_scoring_operation_digest(
    *,
    claim_id: str,
    evaluation_id: str,
    scoring_request_digest: str,
    provider_binding_digest: str,
) -> str:
    """Derive the stable external idempotency commitment for one claim."""

    return canonical_digest(
        {
            "type": "tsq.provider_scoring_operation",
            "claim_id": _require_id(claim_id, "claim_id"),
            "evaluation_id": _require_id(evaluation_id, "evaluation_id"),
            "scoring_request_digest": _require_digest(
                scoring_request_digest, "scoring_request_digest"
            ),
            "provider_binding_digest": _require_digest(
                provider_binding_digest, "provider_binding_digest"
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class ScoringReconciliationRequest:
    """Immutable claim boundary supplied to an observational adapter."""

    claim_id: str
    attempt_id: str
    evaluation_id: str
    through_sequence: int
    provider_id: str
    provider_version: str
    action_trace_digest: str
    command_hash: str
    scoring_request_digest: str
    provider_binding_digest: str
    provider_operation_digest: str
    schema_version: int = RECONCILIATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_id(self.claim_id, "ScoringReconciliationRequest.claim_id")
        _require_id(self.attempt_id, "ScoringReconciliationRequest.attempt_id")
        _require_id(
            self.evaluation_id, "ScoringReconciliationRequest.evaluation_id"
        )
        _require_nonnegative_int(
            self.through_sequence,
            "ScoringReconciliationRequest.through_sequence",
        )
        _require_id(self.provider_id, "ScoringReconciliationRequest.provider_id")
        _require_id(
            self.provider_version,
            "ScoringReconciliationRequest.provider_version",
        )
        _require_digest(
            self.action_trace_digest,
            "ScoringReconciliationRequest.action_trace_digest",
        )
        _require_digest(
            self.command_hash, "ScoringReconciliationRequest.command_hash"
        )
        _require_digest(
            self.scoring_request_digest,
            "ScoringReconciliationRequest.scoring_request_digest",
        )
        _require_digest(
            self.provider_binding_digest,
            "ScoringReconciliationRequest.provider_binding_digest",
        )
        _require_digest(
            self.provider_operation_digest,
            "ScoringReconciliationRequest.provider_operation_digest",
        )
        if self.provider_operation_digest != provider_scoring_operation_digest(
            claim_id=self.claim_id,
            evaluation_id=self.evaluation_id,
            scoring_request_digest=self.scoring_request_digest,
            provider_binding_digest=self.provider_binding_digest,
        ):
            raise ReconciliationProtocolError(
                "ScoringReconciliationRequest.provider_operation_digest does "
                "not match its claim/request/provider commitment."
            )
        if self.schema_version != RECONCILIATION_SCHEMA_VERSION:
            raise ReconciliationProtocolError(
                "ScoringReconciliationRequest.schema_version must be "
                f"{RECONCILIATION_SCHEMA_VERSION}."
            )

    @property
    def digest(self) -> str:
        return canonical_digest(self.terms())

    def terms(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "attempt_id": self.attempt_id,
            "evaluation_id": self.evaluation_id,
            "through_sequence": self.through_sequence,
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "action_trace_digest": self.action_trace_digest,
            "command_hash": self.command_hash,
            "scoring_request_digest": self.scoring_request_digest,
            "provider_binding_digest": self.provider_binding_digest,
            "provider_operation_digest": self.provider_operation_digest,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_terms(cls, value: object) -> "ScoringReconciliationRequest":
        if type(value) is not dict:
            raise ReconciliationProtocolError(
                "Scoring reconciliation request must be an object."
            )
        _require_exact_fields(
            value, _REQUEST_FIELDS, "Scoring reconciliation request"
        )
        request = cls(
            claim_id=value["claim_id"],
            attempt_id=value["attempt_id"],
            evaluation_id=value["evaluation_id"],
            through_sequence=value["through_sequence"],
            provider_id=value["provider_id"],
            provider_version=value["provider_version"],
            action_trace_digest=value["action_trace_digest"],
            command_hash=value["command_hash"],
            scoring_request_digest=value["scoring_request_digest"],
            provider_binding_digest=value["provider_binding_digest"],
            provider_operation_digest=value["provider_operation_digest"],
            schema_version=value["schema_version"],
        )
        if canonical_json(request.terms()) != canonical_json(value):
            raise ReconciliationProtocolError(
                "Scoring reconciliation request is not canonical."
            )
        return request


@dataclass(frozen=True, slots=True)
class ScoringReconciliationReceipt:
    """Canonical provider observation committed to one exact scoring claim."""

    claim_id: str
    attempt_id: str
    evaluation_id: str
    through_sequence: int
    provider_id: str
    provider_version: str
    reconciler_id: str
    reconciler_version: str
    action_trace_digest: str
    command_hash: str
    scoring_request_digest: str
    provider_binding_digest: str
    outcome: ReconciliationOutcome
    observed_at: str
    completed_at: str | None
    result_digest: str | None
    reason_code: str
    provider_operation_digest: str
    provider_receipt_digest: str
    attestation_digest: str
    schema_version: int = RECONCILIATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_id(self.claim_id, "ScoringReconciliationReceipt.claim_id")
        _require_id(self.attempt_id, "ScoringReconciliationReceipt.attempt_id")
        _require_id(
            self.evaluation_id, "ScoringReconciliationReceipt.evaluation_id"
        )
        _require_nonnegative_int(
            self.through_sequence,
            "ScoringReconciliationReceipt.through_sequence",
        )
        _require_id(self.provider_id, "ScoringReconciliationReceipt.provider_id")
        _require_id(
            self.provider_version,
            "ScoringReconciliationReceipt.provider_version",
        )
        _require_id(
            self.reconciler_id, "ScoringReconciliationReceipt.reconciler_id"
        )
        _require_id(
            self.reconciler_version,
            "ScoringReconciliationReceipt.reconciler_version",
        )
        _require_digest(
            self.action_trace_digest,
            "ScoringReconciliationReceipt.action_trace_digest",
        )
        _require_digest(
            self.command_hash, "ScoringReconciliationReceipt.command_hash"
        )
        _require_digest(
            self.scoring_request_digest,
            "ScoringReconciliationReceipt.scoring_request_digest",
        )
        _require_digest(
            self.provider_binding_digest,
            "ScoringReconciliationReceipt.provider_binding_digest",
        )
        if not isinstance(self.outcome, ReconciliationOutcome):
            raise ReconciliationProtocolError(
                "ScoringReconciliationReceipt.outcome must be a "
                "ReconciliationOutcome."
            )
        _canonical_timestamp(
            self.observed_at, "ScoringReconciliationReceipt.observed_at"
        )
        if self.completed_at is not None:
            _canonical_timestamp(
                self.completed_at,
                "ScoringReconciliationReceipt.completed_at",
            )
        _require_id(
            self.reason_code, "ScoringReconciliationReceipt.reason_code"
        )
        _require_digest(
            self.provider_operation_digest,
            "ScoringReconciliationReceipt.provider_operation_digest",
        )
        if self.provider_operation_digest != provider_scoring_operation_digest(
            claim_id=self.claim_id,
            evaluation_id=self.evaluation_id,
            scoring_request_digest=self.scoring_request_digest,
            provider_binding_digest=self.provider_binding_digest,
        ):
            raise ReconciliationProtocolError(
                "ScoringReconciliationReceipt.provider_operation_digest does "
                "not match its claim/request/provider commitment."
            )
        _require_digest(
            self.provider_receipt_digest,
            "ScoringReconciliationReceipt.provider_receipt_digest",
        )
        _require_digest(
            self.attestation_digest,
            "ScoringReconciliationReceipt.attestation_digest",
        )
        if self.outcome is ReconciliationOutcome.COMPLETED:
            if self.completed_at is None or self.result_digest is None:
                raise ReconciliationProtocolError(
                    "A completed reconciliation receipt requires completed_at "
                    "and result_digest."
                )
            _require_digest(
                self.result_digest,
                "ScoringReconciliationReceipt.result_digest",
            )
            if _timestamp_value(self.completed_at) > _timestamp_value(
                self.observed_at
            ):
                raise ReconciliationProtocolError(
                    "A reconciliation cannot observe completion before it occurred."
                )
        elif self.completed_at is not None or self.result_digest is not None:
            raise ReconciliationProtocolError(
                "Unknown and definitely-absent receipts forbid completed_at "
                "and result_digest."
            )
        if self.schema_version != RECONCILIATION_SCHEMA_VERSION:
            raise ReconciliationProtocolError(
                "ScoringReconciliationReceipt.schema_version must be "
                f"{RECONCILIATION_SCHEMA_VERSION}."
            )

    @property
    def digest(self) -> str:
        return canonical_digest(self.terms())

    def terms(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "attempt_id": self.attempt_id,
            "evaluation_id": self.evaluation_id,
            "through_sequence": self.through_sequence,
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "reconciler_id": self.reconciler_id,
            "reconciler_version": self.reconciler_version,
            "action_trace_digest": self.action_trace_digest,
            "command_hash": self.command_hash,
            "scoring_request_digest": self.scoring_request_digest,
            "provider_binding_digest": self.provider_binding_digest,
            "outcome": self.outcome.value,
            "observed_at": self.observed_at,
            "completed_at": self.completed_at,
            "result_digest": self.result_digest,
            "reason_code": self.reason_code,
            "provider_operation_digest": self.provider_operation_digest,
            "provider_receipt_digest": self.provider_receipt_digest,
            "attestation_digest": self.attestation_digest,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_terms(cls, value: object) -> "ScoringReconciliationReceipt":
        if type(value) is not dict:
            raise ReconciliationProtocolError(
                "Scoring reconciliation receipt must be an object."
            )
        _require_exact_fields(
            value, _RECEIPT_FIELDS, "Scoring reconciliation receipt"
        )
        receipt = cls(
            claim_id=value["claim_id"],
            attempt_id=value["attempt_id"],
            evaluation_id=value["evaluation_id"],
            through_sequence=value["through_sequence"],
            provider_id=value["provider_id"],
            provider_version=value["provider_version"],
            reconciler_id=value["reconciler_id"],
            reconciler_version=value["reconciler_version"],
            action_trace_digest=value["action_trace_digest"],
            command_hash=value["command_hash"],
            scoring_request_digest=value["scoring_request_digest"],
            provider_binding_digest=value["provider_binding_digest"],
            outcome=_enum_from_wire(
                value["outcome"],
                ReconciliationOutcome,
                "Scoring reconciliation receipt outcome",
            ),  # type: ignore[arg-type]
            observed_at=value["observed_at"],
            completed_at=value["completed_at"],
            result_digest=value["result_digest"],
            reason_code=value["reason_code"],
            provider_operation_digest=value["provider_operation_digest"],
            provider_receipt_digest=value["provider_receipt_digest"],
            attestation_digest=value["attestation_digest"],
            schema_version=value["schema_version"],
        )
        if canonical_json(receipt.terms()) != canonical_json(value):
            raise ReconciliationProtocolError(
                "Scoring reconciliation receipt is not canonical."
            )
        return receipt

    @classmethod
    def from_json(cls, raw: str) -> "ScoringReconciliationReceipt":
        return cls.from_terms(
            _strict_json_object(raw, "Scoring reconciliation receipt")
        )


@dataclass(frozen=True, slots=True)
class ReconciliationAuthorityBinding:
    """Application trust for one exact observational adapter version.

    ``can_prove_absence`` means the registered adapter is trusted to distinguish
    a durable, tombstoned non-execution from an ambiguous not-found response.
    It grants no authority over rubric scores or learner skill.
    """

    provider_id: str
    provider_version: str
    reconciler_id: str
    reconciler_version: str
    manifest_digest: str
    synthetic: bool = False
    can_prove_absence: bool = False
    schema_version: int = RECONCILIATION_AUTHORITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_id(self.provider_id, "ReconciliationAuthorityBinding.provider_id")
        _require_id(
            self.provider_version,
            "ReconciliationAuthorityBinding.provider_version",
        )
        _require_id(
            self.reconciler_id,
            "ReconciliationAuthorityBinding.reconciler_id",
        )
        _require_id(
            self.reconciler_version,
            "ReconciliationAuthorityBinding.reconciler_version",
        )
        _require_digest(
            self.manifest_digest,
            "ReconciliationAuthorityBinding.manifest_digest",
        )
        _require_bool(
            self.synthetic, "ReconciliationAuthorityBinding.synthetic"
        )
        _require_bool(
            self.can_prove_absence,
            "ReconciliationAuthorityBinding.can_prove_absence",
        )
        if self.synthetic and not self.reconciler_id.startswith("synthetic."):
            raise ReconciliationProtocolError(
                "Synthetic reconciler IDs must start with 'synthetic.'."
            )
        if self.reconciler_id.startswith("synthetic.") and not self.synthetic:
            raise ReconciliationProtocolError(
                "Reconcilers in the synthetic namespace must be marked synthetic."
            )
        if self.schema_version != RECONCILIATION_AUTHORITY_SCHEMA_VERSION:
            raise ReconciliationProtocolError(
                "ReconciliationAuthorityBinding.schema_version must be "
                f"{RECONCILIATION_AUTHORITY_SCHEMA_VERSION}."
            )

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (
            self.provider_id,
            self.provider_version,
            self.reconciler_id,
            self.reconciler_version,
        )

    @property
    def digest(self) -> str:
        return canonical_digest(self.terms())

    def terms(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "reconciler_id": self.reconciler_id,
            "reconciler_version": self.reconciler_version,
            "manifest_digest": self.manifest_digest,
            "synthetic": self.synthetic,
            "can_prove_absence": self.can_prove_absence,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_terms(cls, value: object) -> "ReconciliationAuthorityBinding":
        if type(value) is not dict:
            raise ReconciliationProtocolError(
                "Reconciliation authority binding must be an object."
            )
        _require_exact_fields(
            value, _AUTHORITY_FIELDS, "Reconciliation authority binding"
        )
        binding = cls(
            provider_id=value["provider_id"],
            provider_version=value["provider_version"],
            reconciler_id=value["reconciler_id"],
            reconciler_version=value["reconciler_version"],
            manifest_digest=value["manifest_digest"],
            synthetic=value["synthetic"],
            can_prove_absence=value["can_prove_absence"],
            schema_version=value["schema_version"],
        )
        if canonical_json(binding.terms()) != canonical_json(value):
            raise ReconciliationProtocolError(
                "Reconciliation authority binding is not canonical."
            )
        return binding


@dataclass(frozen=True, slots=True)
class RegisteredReconciler:
    """Stable inspection view of one trusted observational adapter."""

    provider_id: str
    provider_version: str
    reconciler_id: str
    reconciler_version: str
    manifest_digest: str
    binding_digest: str
    synthetic: bool
    can_prove_absence: bool

    def __post_init__(self) -> None:
        binding = ReconciliationAuthorityBinding(
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            reconciler_id=self.reconciler_id,
            reconciler_version=self.reconciler_version,
            manifest_digest=self.manifest_digest,
            synthetic=self.synthetic,
            can_prove_absence=self.can_prove_absence,
        )
        _require_digest(
            self.binding_digest, "RegisteredReconciler.binding_digest"
        )
        if self.binding_digest != binding.digest:
            raise ReconciliationProtocolError(
                "RegisteredReconciler binding digest does not match."
            )

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (
            self.provider_id,
            self.provider_version,
            self.reconciler_id,
            self.reconciler_version,
        )

    def terms(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "reconciler_id": self.reconciler_id,
            "reconciler_version": self.reconciler_version,
            "manifest_digest": self.manifest_digest,
            "binding_digest": self.binding_digest,
            "synthetic": self.synthetic,
            "can_prove_absence": self.can_prove_absence,
            "observational_only": True,
            "skill_authority": False,
        }

    @classmethod
    def from_terms(cls, value: object) -> "RegisteredReconciler":
        if type(value) is not dict:
            raise ReconciliationProtocolError(
                "Registered reconciler must be an object."
            )
        _require_exact_fields(value, _REGISTERED_FIELDS, "Registered reconciler")
        if value["observational_only"] is not True:
            raise ReconciliationProtocolError(
                "Registered reconciler must remain observational-only."
            )
        if value["skill_authority"] is not False:
            raise ReconciliationProtocolError(
                "Registered reconciler cannot claim skill authority."
            )
        reconciler = cls(
            provider_id=value["provider_id"],
            provider_version=value["provider_version"],
            reconciler_id=value["reconciler_id"],
            reconciler_version=value["reconciler_version"],
            manifest_digest=value["manifest_digest"],
            binding_digest=value["binding_digest"],
            synthetic=value["synthetic"],
            can_prove_absence=value["can_prove_absence"],
        )
        if canonical_json(reconciler.terms()) != canonical_json(value):
            raise ReconciliationProtocolError(
                "Registered reconciler is not canonical."
            )
        return reconciler


@dataclass(frozen=True, slots=True)
class ReconciliationObservation:
    """One adapter receipt and its optional authority-free result."""

    receipt: ScoringReconciliationReceipt
    imported_evaluation: ImportedEvaluation | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, ScoringReconciliationReceipt):
            raise ReconciliationProtocolError(
                "ReconciliationObservation.receipt must be a "
                "ScoringReconciliationReceipt."
            )
        if self.receipt.outcome is ReconciliationOutcome.COMPLETED:
            if not isinstance(self.imported_evaluation, ImportedEvaluation):
                raise ReconciliationProtocolError(
                    "A completed reconciliation requires an ImportedEvaluation."
                )
            if self.imported_evaluation.digest != self.receipt.result_digest:
                raise ReconciliationProtocolError(
                    "Completed reconciliation result digest does not match "
                    "its receipt."
                )
        elif self.imported_evaluation is not None:
            raise ReconciliationProtocolError(
                "Non-completed reconciliation outcomes forbid a result."
            )

    @property
    def digest(self) -> str:
        return canonical_digest(self.terms())

    def terms(self) -> dict[str, Any]:
        return {
            "receipt": self.receipt.terms(),
            "imported_evaluation": (
                None
                if self.imported_evaluation is None
                else self.imported_evaluation.terms()
            ),
        }

    @classmethod
    def from_terms(cls, value: object) -> "ReconciliationObservation":
        if type(value) is not dict:
            raise ReconciliationProtocolError(
                "Reconciliation observation must be an object."
            )
        _require_exact_fields(
            value, _OBSERVATION_FIELDS, "Reconciliation observation"
        )
        imported_terms = value["imported_evaluation"]
        try:
            imported = (
                None
                if imported_terms is None
                else ImportedEvaluation.from_terms(imported_terms)
            )
        except ValueError as exc:
            raise ReconciliationProtocolError(
                f"Reconciliation imported evaluation is invalid: {exc}"
            ) from exc
        observation = cls(
            receipt=ScoringReconciliationReceipt.from_terms(value["receipt"]),
            imported_evaluation=imported,
        )
        if canonical_json(observation.terms()) != canonical_json(value):
            raise ReconciliationProtocolError(
                "Reconciliation observation is not canonical."
            )
        return observation


def _validate_reconciliation_binding(
    request: ScoringReconciliationRequest,
    observation: ReconciliationObservation,
    reconciler: RegisteredReconciler,
) -> None:
    """Recompute every request, authority, guarantee, and result commitment."""

    receipt = observation.receipt
    request_boundary = (
        request.claim_id,
        request.attempt_id,
        request.evaluation_id,
        request.through_sequence,
        request.provider_id,
        request.provider_version,
        request.action_trace_digest,
        request.command_hash,
        request.scoring_request_digest,
        request.provider_binding_digest,
        request.provider_operation_digest,
    )
    receipt_boundary = (
        receipt.claim_id,
        receipt.attempt_id,
        receipt.evaluation_id,
        receipt.through_sequence,
        receipt.provider_id,
        receipt.provider_version,
        receipt.action_trace_digest,
        receipt.command_hash,
        receipt.scoring_request_digest,
        receipt.provider_binding_digest,
        receipt.provider_operation_digest,
    )
    if receipt_boundary != request_boundary:
        raise ReconciliationProtocolError(
            "Reconciliation receipt does not match the exact claim request."
        )
    if (
        receipt.reconciler_id != reconciler.reconciler_id
        or receipt.reconciler_version != reconciler.reconciler_version
        or receipt.provider_id != reconciler.provider_id
        or receipt.provider_version != reconciler.provider_version
    ):
        raise ReconciliationProtocolError(
            "Reconciliation receipt does not match its registered authority."
        )
    if (
        receipt.outcome is ReconciliationOutcome.DEFINITELY_ABSENT
        and not reconciler.can_prove_absence
    ):
        raise ReconciliationProtocolError(
            "Registered reconciler cannot prove definite non-execution; "
            "the observation must remain unknown."
        )
    imported = observation.imported_evaluation
    if receipt.outcome is ReconciliationOutcome.COMPLETED:
        if not isinstance(imported, ImportedEvaluation):
            raise ReconciliationProtocolError(
                "Completed reconciliation lacks an ImportedEvaluation."
            )
        canonical = ImportedEvaluation.from_terms(imported.terms())
        if (
            canonical_json(canonical.terms())
            != canonical_json(imported.terms())
            or canonical.digest != receipt.result_digest
        ):
            raise ReconciliationProtocolError(
                "Completed reconciliation result is not canonical or does "
                "not match the receipt digest."
            )
    elif imported is not None:
        raise ReconciliationProtocolError(
            "Non-completed reconciliation outcomes forbid a result."
        )


@runtime_checkable
class TaskScoringReconciler(Protocol):
    """Read-only status lookup for an already admitted provider operation."""

    provider_id: str
    provider_version: str
    reconciler_id: str
    reconciler_version: str
    synthetic: bool
    can_prove_absence: bool

    def lookup(
        self, request: ScoringReconciliationRequest
    ) -> ReconciliationObservation:
        """Observe an existing operation without invoking or retrying scoring."""


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    """Normalized observational result with its configured trust binding."""

    request: ScoringReconciliationRequest
    observation: ReconciliationObservation
    reconciler: RegisteredReconciler
    schema_version: int = RECONCILIATION_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.request, ScoringReconciliationRequest):
            raise ReconciliationProtocolError(
                "ReconciliationResult.request must be a "
                "ScoringReconciliationRequest."
            )
        if not isinstance(self.observation, ReconciliationObservation):
            raise ReconciliationProtocolError(
                "ReconciliationResult.observation must be a "
                "ReconciliationObservation."
            )
        if not isinstance(self.reconciler, RegisteredReconciler):
            raise ReconciliationProtocolError(
                "ReconciliationResult.reconciler must be a RegisteredReconciler."
            )
        if self.schema_version != RECONCILIATION_RESULT_SCHEMA_VERSION:
            raise ReconciliationProtocolError(
                "ReconciliationResult.schema_version must be "
                f"{RECONCILIATION_RESULT_SCHEMA_VERSION}."
            )
        _validate_reconciliation_binding(
            self.request,
            self.observation,
            self.reconciler,
        )

    @property
    def outcome(self) -> ReconciliationOutcome:
        return self.observation.receipt.outcome

    @property
    def imported_evaluation(self) -> ImportedEvaluation | None:
        return self.observation.imported_evaluation

    @property
    def digest(self) -> str:
        return canonical_digest(self.terms())

    def terms(self) -> dict[str, Any]:
        return {
            "request": self.request.terms(),
            "observation": self.observation.terms(),
            "reconciler": self.reconciler.terms(),
            "schema_version": self.schema_version,
            "observational_only": True,
            "automatic_retry_allowed": False,
            "projection_applied": False,
            "certification_applied": False,
            "skill_authority": False,
            "cryptographic_verification_claim": False,
            "attestation_semantics": "registered_adapter_commitment",
        }

    @classmethod
    def from_terms(cls, value: object) -> "ReconciliationResult":
        if type(value) is not dict:
            raise ReconciliationProtocolError(
                "Reconciliation result must be an object."
            )
        _require_exact_fields(value, _RESULT_FIELDS, "Reconciliation result")
        fixed_terms = {
            "observational_only": True,
            "automatic_retry_allowed": False,
            "projection_applied": False,
            "certification_applied": False,
            "skill_authority": False,
            "cryptographic_verification_claim": False,
            "attestation_semantics": "registered_adapter_commitment",
        }
        for field, expected in fixed_terms.items():
            if value[field] is not expected and value[field] != expected:
                raise ReconciliationProtocolError(
                    f"Reconciliation result {field} is invalid."
                )
            if type(expected) is bool and type(value[field]) is not bool:
                raise ReconciliationProtocolError(
                    f"Reconciliation result {field} must be bool."
                )
        result = cls(
            request=ScoringReconciliationRequest.from_terms(value["request"]),
            observation=ReconciliationObservation.from_terms(
                value["observation"]
            ),
            reconciler=RegisteredReconciler.from_terms(value["reconciler"]),
            schema_version=value["schema_version"],
        )
        if canonical_json(result.terms()) != canonical_json(value):
            raise ReconciliationProtocolError(
                "Reconciliation result is not canonical."
            )
        return result


@dataclass(frozen=True, slots=True)
class _ReconcilerRegistration:
    adapter: TaskScoringReconciler
    binding: ReconciliationAuthorityBinding
    summary: RegisteredReconciler


class ScoringReconciliationRegistry:
    """Exact-version trust registry for observational reconciliation adapters."""

    def __init__(self, *, allow_synthetic: bool = False) -> None:
        _require_bool(allow_synthetic, "allow_synthetic")
        self._allow_synthetic = allow_synthetic
        self._reconcilers: dict[
            tuple[str, str, str, str], _ReconcilerRegistration
        ] = {}

    @staticmethod
    def _adapter_identity(
        adapter: TaskScoringReconciler,
    ) -> tuple[str, str, str, str, bool, bool]:
        try:
            identity = (
                adapter.provider_id,
                adapter.provider_version,
                adapter.reconciler_id,
                adapter.reconciler_version,
                adapter.synthetic,
                adapter.can_prove_absence,
            )
        except (AttributeError, TypeError) as exc:
            raise ReconciliationProtocolError(
                "Adapter does not expose the reconciliation identity."
            ) from exc
        _require_id(identity[0], "adapter.provider_id")
        _require_id(identity[1], "adapter.provider_version")
        _require_id(identity[2], "adapter.reconciler_id")
        _require_id(identity[3], "adapter.reconciler_version")
        _require_bool(identity[4], "adapter.synthetic")
        _require_bool(identity[5], "adapter.can_prove_absence")
        if callable(getattr(adapter, "score", None)):
            raise ReconciliationProtocolError(
                "Observational reconciliation adapters must not expose a "
                "score method."
            )
        return identity

    def register(
        self,
        adapter: TaskScoringReconciler,
        binding: ReconciliationAuthorityBinding,
    ) -> RegisteredReconciler:
        """Register one exact observational adapter without scorer authority."""

        if not isinstance(binding, ReconciliationAuthorityBinding):
            raise ReconciliationProtocolError(
                "binding must be a ReconciliationAuthorityBinding."
            )
        identity = self._adapter_identity(adapter)
        try:
            lookup_method = adapter.lookup
        except AttributeError as exc:
            raise ReconciliationProtocolError(
                "Adapter does not implement observational lookup."
            ) from exc
        if not callable(lookup_method):
            raise ReconciliationProtocolError("adapter.lookup must be callable.")
        if callable(getattr(adapter, "score", None)):
            raise ReconciliationProtocolError(
                "Observational reconciliation adapters must not expose a "
                "score method."
            )
        expected_identity = (
            binding.provider_id,
            binding.provider_version,
            binding.reconciler_id,
            binding.reconciler_version,
            binding.synthetic,
            binding.can_prove_absence,
        )
        if identity != expected_identity:
            raise ReconciliationProtocolError(
                "Adapter identity does not match its reconciliation authority."
            )
        if binding.synthetic and not self._allow_synthetic:
            raise ReconciliationProtocolError(
                "Synthetic reconcilers require allow_synthetic=True."
            )
        if binding.key in self._reconcilers:
            raise ReconciliationProtocolError(
                "Reconciler "
                f"{binding.reconciler_id}@{binding.reconciler_version} for "
                f"{binding.provider_id}@{binding.provider_version} is already "
                "registered."
            )
        summary = RegisteredReconciler(
            provider_id=binding.provider_id,
            provider_version=binding.provider_version,
            reconciler_id=binding.reconciler_id,
            reconciler_version=binding.reconciler_version,
            manifest_digest=binding.manifest_digest,
            binding_digest=binding.digest,
            synthetic=binding.synthetic,
            can_prove_absence=binding.can_prove_absence,
        )
        self._reconcilers[binding.key] = _ReconcilerRegistration(
            adapter=adapter,
            binding=binding,
            summary=summary,
        )
        return summary

    def list(self) -> tuple[RegisteredReconciler, ...]:
        return tuple(
            self._reconcilers[key].summary
            for key in sorted(self._reconcilers)
        )

    def inspect(
        self,
        provider_id: str,
        provider_version: str,
        reconciler_id: str,
        reconciler_version: str,
    ) -> RegisteredReconciler:
        key = (
            _require_id(provider_id, "provider_id"),
            _require_id(provider_version, "provider_version"),
            _require_id(reconciler_id, "reconciler_id"),
            _require_id(reconciler_version, "reconciler_version"),
        )
        try:
            return self._reconcilers[key].summary
        except KeyError as exc:
            raise ReconcilerNotFoundError(
                "Reconciler "
                f"{reconciler_id}@{reconciler_version} for "
                f"{provider_id}@{provider_version} is not registered."
            ) from exc

    def reconcile(
        self,
        reconciler_id: str,
        reconciler_version: str,
        request: ScoringReconciliationRequest,
    ) -> ReconciliationResult:
        """Look up and normalize an observation without invoking scoring."""

        if not isinstance(request, ScoringReconciliationRequest):
            raise ReconciliationProtocolError(
                "request must be a ScoringReconciliationRequest."
            )
        summary = self.inspect(
            request.provider_id,
            request.provider_version,
            reconciler_id,
            reconciler_version,
        )
        registration = self._reconcilers[summary.key]
        expected_identity = (
            summary.provider_id,
            summary.provider_version,
            summary.reconciler_id,
            summary.reconciler_version,
            summary.synthetic,
            summary.can_prove_absence,
        )
        try:
            current_identity = self._adapter_identity(registration.adapter)
        except ReconciliationProtocolError as exc:
            raise ReconcilerExecutionError(
                "Registered reconciler identity became unavailable."
            ) from exc
        if current_identity != expected_identity:
            raise ReconcilerExecutionError(
                "Registered reconciler identity changed after registration."
            )
        try:
            observation = registration.adapter.lookup(request)
        except Exception as exc:
            raise ReconcilerExecutionError(
                "Registered reconciliation lookup failed."
            ) from exc
        try:
            final_identity = self._adapter_identity(registration.adapter)
        except ReconciliationProtocolError as exc:
            raise ReconcilerExecutionError(
                "Registered reconciler identity became unavailable after lookup."
            ) from exc
        if final_identity != expected_identity:
            raise ReconcilerExecutionError(
                "Registered reconciler identity changed during lookup."
            )
        if not isinstance(observation, ReconciliationObservation):
            raise ReconcilerExecutionError(
                "Registered reconciler returned a non-protocol observation."
            )
        self._validate_observation(request, observation, summary)
        return ReconciliationResult(
            request=request,
            observation=observation,
            reconciler=summary,
        )

    @staticmethod
    def _validate_observation(
        request: ScoringReconciliationRequest,
        observation: ReconciliationObservation,
        reconciler: RegisteredReconciler,
    ) -> None:
        _validate_reconciliation_binding(request, observation, reconciler)


class SyntheticReconciliationAdapter:
    """Deterministic test adapter for one immutable observational fixture."""

    synthetic = True

    def __init__(
        self,
        observation: ReconciliationObservation,
        *,
        reconciler_id: str = "synthetic.fixed-reconciler",
        reconciler_version: str = "test-v1",
        can_prove_absence: bool = False,
    ) -> None:
        if not isinstance(observation, ReconciliationObservation):
            raise ReconciliationProtocolError(
                "Synthetic adapter observation must be a "
                "ReconciliationObservation."
            )
        _require_id(reconciler_id, "Synthetic adapter reconciler_id")
        if not reconciler_id.startswith("synthetic."):
            raise ReconciliationProtocolError(
                "Synthetic reconciler IDs must start with 'synthetic.'."
            )
        _require_id(
            reconciler_version, "Synthetic adapter reconciler_version"
        )
        _require_bool(can_prove_absence, "can_prove_absence")
        self.provider_id = observation.receipt.provider_id
        self.provider_version = observation.receipt.provider_version
        self.reconciler_id = reconciler_id
        self.reconciler_version = reconciler_version
        self.can_prove_absence = can_prove_absence
        self._observation = observation
        self.lookup_calls = 0

    @property
    def authority_binding(self) -> ReconciliationAuthorityBinding:
        manifest_digest = canonical_digest(
            {
                "provider_id": self.provider_id,
                "provider_version": self.provider_version,
                "reconciler_id": self.reconciler_id,
                "reconciler_version": self.reconciler_version,
                "can_prove_absence": self.can_prove_absence,
                "synthetic": True,
                "observational_only": True,
            }
        )
        return ReconciliationAuthorityBinding(
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            reconciler_id=self.reconciler_id,
            reconciler_version=self.reconciler_version,
            manifest_digest=manifest_digest,
            synthetic=True,
            can_prove_absence=self.can_prove_absence,
        )

    def lookup(
        self, request: ScoringReconciliationRequest
    ) -> ReconciliationObservation:
        if not isinstance(request, ScoringReconciliationRequest):
            raise ReconciliationProtocolError(
                "request must be a ScoringReconciliationRequest."
            )
        self.lookup_calls += 1
        return self._observation


__all__ = [
    "RECONCILIATION_AUTHORITY_SCHEMA_VERSION",
    "RECONCILIATION_RESULT_SCHEMA_VERSION",
    "RECONCILIATION_SCHEMA_VERSION",
    "ReconcilerExecutionError",
    "ReconcilerNotFoundError",
    "ReconciliationAuthorityBinding",
    "ReconciliationObservation",
    "ReconciliationOutcome",
    "ReconciliationProtocolError",
    "ReconciliationResult",
    "RegisteredReconciler",
    "ScoringReconciliationReceipt",
    "ScoringReconciliationRegistry",
    "ScoringReconciliationRequest",
    "SyntheticReconciliationAdapter",
    "TaskScoringReconciler",
    "provider_scoring_operation_digest",
]
