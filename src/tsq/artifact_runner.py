# SPDX-License-Identifier: MPL-2.0

"""Process-separated checking of one inert, bounded artifact format.

This module is deliberately much smaller than a general artifact executor.  It
accepts bytes already captured by a caller, binds them to a canonical request,
and sends them over stdin to one fixed, bundled checker.  The checker parses a
small UTF-8 JSON causal-mask matrix as data.  Learner bytes are never used as a
path, module, command, callback, import, or executable program.

The subprocess boundary is *not* an operating-system sandbox.  Python in the
child retains the ambient filesystem and network authority of the current
account.  Only trusted bundled checker code is run, so this module must never
be extended to execute learner-authored Python, shell, archives, or plugins
without a separately reviewed operating-system isolation boundary.

The registry is synthetic-only and confers no scoring, mastery, projection, or
certification authority.  Its receipt records those limits explicitly.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, BinaryIO

from .evidence import canonical_digest, canonical_json


ARTIFACT_RUNNER_SCHEMA_VERSION = 1
ARTIFACT_WORKER_PROTOCOL_VERSION = 1
MAX_RUNNER_ARTIFACT_BYTES = 65_536
MAX_WORKER_OUTPUT_BYTES = 16_384
DEFAULT_WORKER_TIMEOUT_MS = 2_000
MAX_BUNDLED_WORKER_BYTES = 256 * 1024

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ArtifactRunnerProtocolError(ValueError):
    """A request, binding, result, or receipt violated the closed protocol."""


class ArtifactRunnerNotFoundError(LookupError):
    """No exact checker ID/version was registered."""


class ArtifactCheckerId(StrEnum):
    """Closed checkers shipped by this deliberately synthetic slice."""

    CAUSAL_MASK_MATRIX = "synthetic.causal-mask-matrix"


class ArtifactRunOutcome(StrEnum):
    """Closed host-visible outcomes for one admitted pure check."""

    COMPLETED = "completed"
    INVALID_ARTIFACT = "invalid_artifact"
    TIMED_OUT = "timed_out"
    WORKER_FAILED = "worker_failed"
    PROTOCOL_ERROR = "protocol_error"


class ArtifactResultCode(StrEnum):
    """Closed, non-prose observations emitted by the worker or host."""

    MATRIX_SHAPE_VALID = "matrix_shape_valid"
    CAUSAL_VISIBILITY_VALID = "causal_visibility_valid"
    CAUSAL_VISIBILITY_INVALID = "causal_visibility_invalid"
    EMPTY_INPUT = "empty_input"
    INPUT_TOO_LARGE = "input_too_large"
    INVALID_UTF8 = "invalid_utf8"
    JSON_DEPTH_EXCEEDED = "json_depth_exceeded"
    INVALID_JSON = "invalid_json"
    DUPLICATE_FIELD = "duplicate_field"
    NONFINITE_NUMBER = "nonfinite_number"
    INVALID_DOCUMENT = "invalid_document"
    UNKNOWN_OR_MISSING_FIELDS = "unknown_or_missing_fields"
    INVALID_SCHEMA_VERSION = "invalid_schema_version"
    INVALID_MATRIX = "invalid_matrix"
    ARTIFACT_DIGEST_MISMATCH = "artifact_digest_mismatch"
    WORKER_TIMEOUT = "worker_timeout"
    WORKER_START_FAILED = "worker_start_failed"
    WORKER_EXIT_NONZERO = "worker_exit_nonzero"
    WORKER_OUTPUT_LIMIT = "worker_output_limit"
    WORKER_STDERR = "worker_stderr"
    WORKER_PROTOCOL_INVALID = "worker_protocol_invalid"
    BUNDLED_WORKER_MISMATCH = "bundled_worker_mismatch"


SYNTHETIC_RUNNER_ID = "synthetic.stdlib-process-runner"
SYNTHETIC_RUNNER_VERSION = "v1"
CAUSAL_MASK_ARTIFACT_KIND = "causal_mask_matrix_v1"
CAUSAL_MASK_CHECK_SET_ID = "causal_mask_matrix_checks_v1"
# This is a release commitment, not a digest discovered and trusted at runtime.
# Changing the bundled worker requires a new checker/runner version and manifest.
BUNDLED_WORKER_SHA256 = (
    "2b3e5f04490c6b2ca81b45e06311e91315ebf0fd811ae4ade550b8fecf59503f"
)

CAUSAL_MASK_ARTIFACT_MANIFEST_DIGEST = canonical_digest(
    {
        "type": "tsq.synthetic_artifact_format_manifest",
        "schema_version": 1,
        "artifact_kind": CAUSAL_MASK_ARTIFACT_KIND,
        "media_type": "application/json",
        "document_fields": ["mask", "schema_version"],
        "matrix_cell_type": "boolean",
        "maximum_matrix_size": 64,
        "executable": False,
    }
)
CAUSAL_MASK_CHECK_SET_MANIFEST_DIGEST = canonical_digest(
    {
        "type": "tsq.synthetic_check_set_manifest",
        "schema_version": 1,
        "check_set_id": CAUSAL_MASK_CHECK_SET_ID,
        "checker_id": ArtifactCheckerId.CAUSAL_MASK_MATRIX.value,
        "checker_version": "v1",
        "artifact_kind": CAUSAL_MASK_ARTIFACT_KIND,
        "checks": ["causal_visibility", "matrix_shape"],
        "worker_protocol_version": ARTIFACT_WORKER_PROTOCOL_VERSION,
        "worker_sha256": BUNDLED_WORKER_SHA256,
        "synthetic": True,
    }
)


def _require_exact_fields(
    value: dict[str, Any],
    expected: frozenset[str],
    label: str,
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
    raise ArtifactRunnerProtocolError(f"{label} has " + "; ".join(details) + ".")


def _require_id(value: object, label: str) -> str:
    if type(value) is not str or not _ID_PATTERN.fullmatch(value):
        raise ArtifactRunnerProtocolError(
            f"{label} must be a stable identifier."
        )
    return value


def _require_digest(value: object, label: str) -> str:
    if type(value) is not str or not _DIGEST_PATTERN.fullmatch(value):
        raise ArtifactRunnerProtocolError(
            f"{label} must be a lowercase SHA-256 digest."
        )
    return value


def _require_int(
    value: object,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if (
        type(value) is not int
        or value < minimum
        or value > maximum
    ):
        raise ArtifactRunnerProtocolError(
            f"{label} must be an integer between {minimum} and {maximum}."
        )
    return value


def _enum_value(
    value: object,
    enum_type: type[StrEnum],
    label: str,
) -> StrEnum:
    if type(value) is not str:
        raise ArtifactRunnerProtocolError(f"{label} must be a string.")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ArtifactRunnerProtocolError(
            f"{label} has unknown value {value!r}."
        ) from exc


def _require_bool(value: object, expected: bool, label: str) -> None:
    if type(value) is not bool or value is not expected:
        raise ArtifactRunnerProtocolError(f"{label} must be {expected!r}.")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ArtifactRunnerProtocolError(
                f"JSON contains duplicate field {key!r}."
            )
        value[key] = item
    return value


def _reject_constant(raw: str) -> None:
    raise ArtifactRunnerProtocolError(
        f"JSON contains invalid non-finite number {raw}."
    )


def _reject_float(_raw: str) -> float:
    raise ArtifactRunnerProtocolError(
        "Runner protocol JSON does not permit floating-point numbers."
    )


def _parse_bounded_integer(raw: str) -> int:
    if len(raw) > 16:
        raise ArtifactRunnerProtocolError(
            "Runner protocol JSON contains an oversized integer."
        )
    return int(raw)


def _strict_json_object(
    raw: str | bytes,
    *,
    label: str,
    maximum_bytes: int,
) -> dict[str, Any]:
    if isinstance(raw, bytes):
        if len(raw) > maximum_bytes:
            raise ArtifactRunnerProtocolError(f"{label} exceeds its size limit.")
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ArtifactRunnerProtocolError(
                f"{label} is not valid UTF-8."
            ) from exc
    elif isinstance(raw, str):
        try:
            size = len(raw.encode("utf-8", errors="strict"))
        except UnicodeEncodeError as exc:
            raise ArtifactRunnerProtocolError(
                f"{label} is not valid UTF-8."
            ) from exc
        if size > maximum_bytes:
            raise ArtifactRunnerProtocolError(f"{label} exceeds its size limit.")
        text = raw
    else:
        raise ArtifactRunnerProtocolError(f"{label} must be text or bytes.")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
            parse_float=_reject_float,
            parse_int=_parse_bounded_integer,
        )
    except ArtifactRunnerProtocolError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ArtifactRunnerProtocolError(f"{label} is invalid JSON.") from exc
    if type(value) is not dict:
        raise ArtifactRunnerProtocolError(f"{label} must be a JSON object.")
    return value


@dataclass(frozen=True, slots=True)
class ArtifactRunnerBinding:
    """Immutable application binding for the one bundled synthetic checker."""

    runner_id: str
    runner_version: str
    checker_id: ArtifactCheckerId
    checker_version: str
    artifact_kind: str
    artifact_manifest_digest: str
    check_set_id: str
    check_set_manifest_digest: str
    worker_sha256: str
    maximum_input_bytes: int = MAX_RUNNER_ARTIFACT_BYTES
    maximum_output_bytes: int = MAX_WORKER_OUTPUT_BYTES
    timeout_ms: int = DEFAULT_WORKER_TIMEOUT_MS
    synthetic: bool = True
    process_separated: bool = True
    operating_system_sandboxed: bool = False
    filesystem_isolation_enforced: bool = False
    network_isolation_enforced: bool = False
    schema_version: int = ARTIFACT_RUNNER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_id(self.runner_id, "ArtifactRunnerBinding.runner_id")
        _require_id(self.runner_version, "ArtifactRunnerBinding.runner_version")
        if not isinstance(self.checker_id, ArtifactCheckerId):
            raise ArtifactRunnerProtocolError(
                "ArtifactRunnerBinding.checker_id must be an ArtifactCheckerId."
            )
        _require_id(
            self.checker_version,
            "ArtifactRunnerBinding.checker_version",
        )
        _require_id(self.artifact_kind, "ArtifactRunnerBinding.artifact_kind")
        _require_digest(
            self.artifact_manifest_digest,
            "ArtifactRunnerBinding.artifact_manifest_digest",
        )
        _require_id(self.check_set_id, "ArtifactRunnerBinding.check_set_id")
        _require_digest(
            self.check_set_manifest_digest,
            "ArtifactRunnerBinding.check_set_manifest_digest",
        )
        _require_digest(
            self.worker_sha256,
            "ArtifactRunnerBinding.worker_sha256",
        )
        _require_int(
            self.maximum_input_bytes,
            "ArtifactRunnerBinding.maximum_input_bytes",
            minimum=1,
            maximum=MAX_RUNNER_ARTIFACT_BYTES,
        )
        _require_int(
            self.maximum_output_bytes,
            "ArtifactRunnerBinding.maximum_output_bytes",
            minimum=1_024,
            maximum=MAX_WORKER_OUTPUT_BYTES,
        )
        _require_int(
            self.timeout_ms,
            "ArtifactRunnerBinding.timeout_ms",
            minimum=50,
            maximum=10_000,
        )
        _require_bool(self.synthetic, True, "ArtifactRunnerBinding.synthetic")
        _require_bool(
            self.process_separated,
            True,
            "ArtifactRunnerBinding.process_separated",
        )
        _require_bool(
            self.operating_system_sandboxed,
            False,
            "ArtifactRunnerBinding.operating_system_sandboxed",
        )
        _require_bool(
            self.filesystem_isolation_enforced,
            False,
            "ArtifactRunnerBinding.filesystem_isolation_enforced",
        )
        _require_bool(
            self.network_isolation_enforced,
            False,
            "ArtifactRunnerBinding.network_isolation_enforced",
        )
        _require_int(
            self.schema_version,
            "ArtifactRunnerBinding.schema_version",
            minimum=ARTIFACT_RUNNER_SCHEMA_VERSION,
            maximum=ARTIFACT_RUNNER_SCHEMA_VERSION,
        )

    @property
    def key(self) -> tuple[ArtifactCheckerId, str]:
        return (self.checker_id, self.checker_version)

    @property
    def digest(self) -> str:
        return canonical_digest(self.terms())

    def terms(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "artifact_manifest_digest": self.artifact_manifest_digest,
            "checker_id": self.checker_id.value,
            "checker_version": self.checker_version,
            "check_set_id": self.check_set_id,
            "check_set_manifest_digest": self.check_set_manifest_digest,
            "filesystem_isolation_enforced": self.filesystem_isolation_enforced,
            "maximum_input_bytes": self.maximum_input_bytes,
            "maximum_output_bytes": self.maximum_output_bytes,
            "network_isolation_enforced": self.network_isolation_enforced,
            "operating_system_sandboxed": self.operating_system_sandboxed,
            "process_separated": self.process_separated,
            "runner_id": self.runner_id,
            "runner_version": self.runner_version,
            "schema_version": self.schema_version,
            "synthetic": self.synthetic,
            "timeout_ms": self.timeout_ms,
            "worker_sha256": self.worker_sha256,
        }

    @classmethod
    def from_terms(cls, value: object) -> "ArtifactRunnerBinding":
        if type(value) is not dict:
            raise ArtifactRunnerProtocolError(
                "Artifact runner binding must be an object."
            )
        _require_exact_fields(
            value,
            frozenset(
                {
                    "artifact_kind",
                    "artifact_manifest_digest",
                    "checker_id",
                    "checker_version",
                    "check_set_id",
                    "check_set_manifest_digest",
                    "filesystem_isolation_enforced",
                    "maximum_input_bytes",
                    "maximum_output_bytes",
                    "network_isolation_enforced",
                    "operating_system_sandboxed",
                    "process_separated",
                    "runner_id",
                    "runner_version",
                    "schema_version",
                    "synthetic",
                    "timeout_ms",
                    "worker_sha256",
                }
            ),
            "Artifact runner binding",
        )
        return cls(
            runner_id=value["runner_id"],
            runner_version=value["runner_version"],
            checker_id=_enum_value(
                value["checker_id"],
                ArtifactCheckerId,
                "ArtifactRunnerBinding.checker_id",
            ),
            checker_version=value["checker_version"],
            artifact_kind=value["artifact_kind"],
            artifact_manifest_digest=value["artifact_manifest_digest"],
            check_set_id=value["check_set_id"],
            check_set_manifest_digest=value["check_set_manifest_digest"],
            worker_sha256=value["worker_sha256"],
            maximum_input_bytes=value["maximum_input_bytes"],
            maximum_output_bytes=value["maximum_output_bytes"],
            timeout_ms=value["timeout_ms"],
            synthetic=value["synthetic"],
            process_separated=value["process_separated"],
            operating_system_sandboxed=value["operating_system_sandboxed"],
            filesystem_isolation_enforced=value[
                "filesystem_isolation_enforced"
            ],
            network_isolation_enforced=value["network_isolation_enforced"],
            schema_version=value["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class ArtifactRunRequest:
    """Content-free commitment for one exact checker invocation."""

    run_id: str
    checker_id: ArtifactCheckerId
    checker_version: str
    artifact_kind: str
    artifact_manifest_digest: str
    artifact_sha256: str
    artifact_size_bytes: int
    check_set_id: str
    check_set_manifest_digest: str
    runner_binding_digest: str
    schema_version: int = ARTIFACT_RUNNER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_id(self.run_id, "ArtifactRunRequest.run_id")
        if not isinstance(self.checker_id, ArtifactCheckerId):
            raise ArtifactRunnerProtocolError(
                "ArtifactRunRequest.checker_id must be an ArtifactCheckerId."
            )
        _require_id(self.checker_version, "ArtifactRunRequest.checker_version")
        _require_id(self.artifact_kind, "ArtifactRunRequest.artifact_kind")
        _require_digest(
            self.artifact_manifest_digest,
            "ArtifactRunRequest.artifact_manifest_digest",
        )
        _require_digest(
            self.artifact_sha256,
            "ArtifactRunRequest.artifact_sha256",
        )
        _require_int(
            self.artifact_size_bytes,
            "ArtifactRunRequest.artifact_size_bytes",
            minimum=0,
            maximum=MAX_RUNNER_ARTIFACT_BYTES + 1,
        )
        _require_id(self.check_set_id, "ArtifactRunRequest.check_set_id")
        _require_digest(
            self.check_set_manifest_digest,
            "ArtifactRunRequest.check_set_manifest_digest",
        )
        _require_digest(
            self.runner_binding_digest,
            "ArtifactRunRequest.runner_binding_digest",
        )
        _require_int(
            self.schema_version,
            "ArtifactRunRequest.schema_version",
            minimum=ARTIFACT_RUNNER_SCHEMA_VERSION,
            maximum=ARTIFACT_RUNNER_SCHEMA_VERSION,
        )

    @property
    def digest(self) -> str:
        return canonical_digest(self.terms())

    def terms(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "artifact_manifest_digest": self.artifact_manifest_digest,
            "artifact_sha256": self.artifact_sha256,
            "artifact_size_bytes": self.artifact_size_bytes,
            "checker_id": self.checker_id.value,
            "checker_version": self.checker_version,
            "check_set_id": self.check_set_id,
            "check_set_manifest_digest": self.check_set_manifest_digest,
            "run_id": self.run_id,
            "runner_binding_digest": self.runner_binding_digest,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_terms(cls, value: object) -> "ArtifactRunRequest":
        if type(value) is not dict:
            raise ArtifactRunnerProtocolError(
                "Artifact run request must be an object."
            )
        _require_exact_fields(
            value,
            frozenset(
                {
                    "artifact_kind",
                    "artifact_manifest_digest",
                    "artifact_sha256",
                    "artifact_size_bytes",
                    "checker_id",
                    "checker_version",
                    "check_set_id",
                    "check_set_manifest_digest",
                    "run_id",
                    "runner_binding_digest",
                    "schema_version",
                }
            ),
            "Artifact run request",
        )
        return cls(
            run_id=value["run_id"],
            checker_id=_enum_value(
                value["checker_id"],
                ArtifactCheckerId,
                "ArtifactRunRequest.checker_id",
            ),
            checker_version=value["checker_version"],
            artifact_kind=value["artifact_kind"],
            artifact_manifest_digest=value["artifact_manifest_digest"],
            artifact_sha256=value["artifact_sha256"],
            artifact_size_bytes=value["artifact_size_bytes"],
            check_set_id=value["check_set_id"],
            check_set_manifest_digest=value["check_set_manifest_digest"],
            runner_binding_digest=value["runner_binding_digest"],
            schema_version=value["schema_version"],
        )


_INVALID_ARTIFACT_CODES = frozenset(
    {
        ArtifactResultCode.EMPTY_INPUT,
        ArtifactResultCode.INPUT_TOO_LARGE,
        ArtifactResultCode.INVALID_UTF8,
        ArtifactResultCode.JSON_DEPTH_EXCEEDED,
        ArtifactResultCode.INVALID_JSON,
        ArtifactResultCode.DUPLICATE_FIELD,
        ArtifactResultCode.NONFINITE_NUMBER,
        ArtifactResultCode.INVALID_DOCUMENT,
        ArtifactResultCode.UNKNOWN_OR_MISSING_FIELDS,
        ArtifactResultCode.INVALID_SCHEMA_VERSION,
        ArtifactResultCode.INVALID_MATRIX,
    }
)


@dataclass(frozen=True, slots=True)
class ArtifactRunResult:
    """Closed, prose-free result from the worker or its supervising host."""

    checker_id: ArtifactCheckerId
    checker_version: str
    artifact_sha256: str
    outcome: ArtifactRunOutcome
    outcome_codes: tuple[ArtifactResultCode, ...]
    passed: int
    failed: int
    errored: int
    skipped: int
    schema_version: int = ARTIFACT_RUNNER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.checker_id, ArtifactCheckerId):
            raise ArtifactRunnerProtocolError(
                "ArtifactRunResult.checker_id must be an ArtifactCheckerId."
            )
        _require_id(self.checker_version, "ArtifactRunResult.checker_version")
        _require_digest(
            self.artifact_sha256,
            "ArtifactRunResult.artifact_sha256",
        )
        if not isinstance(self.outcome, ArtifactRunOutcome):
            raise ArtifactRunnerProtocolError(
                "ArtifactRunResult.outcome must be an ArtifactRunOutcome."
            )
        if (
            type(self.outcome_codes) is not tuple
            or any(
                not isinstance(code, ArtifactResultCode)
                for code in self.outcome_codes
            )
        ):
            raise ArtifactRunnerProtocolError(
                "ArtifactRunResult.outcome_codes must contain closed result codes."
            )
        normalized_codes = tuple(
            sorted(set(self.outcome_codes), key=lambda item: item.value)
        )
        if len(normalized_codes) != len(self.outcome_codes):
            raise ArtifactRunnerProtocolError(
                "ArtifactRunResult.outcome_codes must be unique."
            )
        object.__setattr__(self, "outcome_codes", normalized_codes)
        for field_name in ("passed", "failed", "errored", "skipped"):
            _require_int(
                getattr(self, field_name),
                f"ArtifactRunResult.{field_name}",
                minimum=0,
                maximum=1_000_000,
            )
        _require_int(
            self.schema_version,
            "ArtifactRunResult.schema_version",
            minimum=ARTIFACT_RUNNER_SCHEMA_VERSION,
            maximum=ARTIFACT_RUNNER_SCHEMA_VERSION,
        )

        code_set = frozenset(self.outcome_codes)
        counts = (self.passed, self.failed, self.errored, self.skipped)
        valid_completed = (
            (
                code_set
                == {
                    ArtifactResultCode.MATRIX_SHAPE_VALID,
                    ArtifactResultCode.CAUSAL_VISIBILITY_VALID,
                }
                and counts == (2, 0, 0, 0)
            )
            or (
                code_set
                == {
                    ArtifactResultCode.MATRIX_SHAPE_VALID,
                    ArtifactResultCode.CAUSAL_VISIBILITY_INVALID,
                }
                and counts == (1, 1, 0, 0)
            )
        )
        valid_single_error = (
            len(code_set) == 1 and counts == (0, 0, 1, 0)
        )
        if self.outcome is ArtifactRunOutcome.COMPLETED:
            valid = valid_completed
        elif self.outcome is ArtifactRunOutcome.INVALID_ARTIFACT:
            valid = valid_single_error and bool(code_set & _INVALID_ARTIFACT_CODES)
        elif self.outcome is ArtifactRunOutcome.TIMED_OUT:
            valid = (
                valid_single_error
                and code_set == {ArtifactResultCode.WORKER_TIMEOUT}
            )
        elif self.outcome is ArtifactRunOutcome.WORKER_FAILED:
            valid = valid_single_error and code_set in (
                {ArtifactResultCode.WORKER_START_FAILED},
                {ArtifactResultCode.WORKER_EXIT_NONZERO},
            )
        else:
            valid = valid_single_error and code_set in (
                {ArtifactResultCode.WORKER_OUTPUT_LIMIT},
                {ArtifactResultCode.WORKER_STDERR},
                {ArtifactResultCode.WORKER_PROTOCOL_INVALID},
                {ArtifactResultCode.BUNDLED_WORKER_MISMATCH},
            )
        if not valid:
            raise ArtifactRunnerProtocolError(
                "ArtifactRunResult outcome, codes, and counts are inconsistent."
            )

    @property
    def digest(self) -> str:
        return canonical_digest(self.terms())

    def terms(self) -> dict[str, Any]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "checker_id": self.checker_id.value,
            "checker_version": self.checker_version,
            "errored": self.errored,
            "failed": self.failed,
            "outcome": self.outcome.value,
            "outcome_codes": [code.value for code in self.outcome_codes],
            "passed": self.passed,
            "schema_version": self.schema_version,
            "skipped": self.skipped,
        }

    @classmethod
    def from_terms(cls, value: object) -> "ArtifactRunResult":
        if type(value) is not dict:
            raise ArtifactRunnerProtocolError(
                "Artifact run result must be an object."
            )
        _require_exact_fields(
            value,
            frozenset(
                {
                    "artifact_sha256",
                    "checker_id",
                    "checker_version",
                    "errored",
                    "failed",
                    "outcome",
                    "outcome_codes",
                    "passed",
                    "schema_version",
                    "skipped",
                }
            ),
            "Artifact run result",
        )
        raw_codes = value["outcome_codes"]
        if type(raw_codes) is not list:
            raise ArtifactRunnerProtocolError(
                "ArtifactRunResult.outcome_codes must be an array."
            )
        return cls(
            checker_id=_enum_value(
                value["checker_id"],
                ArtifactCheckerId,
                "ArtifactRunResult.checker_id",
            ),
            checker_version=value["checker_version"],
            artifact_sha256=value["artifact_sha256"],
            outcome=_enum_value(
                value["outcome"],
                ArtifactRunOutcome,
                "ArtifactRunResult.outcome",
            ),
            outcome_codes=tuple(
                _enum_value(
                    item,
                    ArtifactResultCode,
                    "ArtifactRunResult.outcome_codes[]",
                )
                for item in raw_codes
            ),
            passed=value["passed"],
            failed=value["failed"],
            errored=value["errored"],
            skipped=value["skipped"],
            schema_version=value["schema_version"],
        )

    @classmethod
    def from_json(cls, raw: str | bytes) -> "ArtifactRunResult":
        return cls.from_terms(
            _strict_json_object(
                raw,
                label="Artifact worker result",
                maximum_bytes=MAX_WORKER_OUTPUT_BYTES,
            )
        )


@dataclass(frozen=True, slots=True)
class ArtifactProcessReceipt:
    """Canonical process observation with explicit non-authority boundaries.

    This is not the operational claim receipt stored by the performance
    ledger. The ledger wraps this process observation with claim, attempt,
    action, and timestamp identities at its own boundary.
    """

    request: ArtifactRunRequest
    binding: ArtifactRunnerBinding
    result: ArtifactRunResult
    worker_process_started: bool
    schema_version: int = ARTIFACT_RUNNER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.request, ArtifactRunRequest):
            raise ArtifactRunnerProtocolError(
                "ArtifactProcessReceipt.request must be an ArtifactRunRequest."
            )
        if not isinstance(self.binding, ArtifactRunnerBinding):
            raise ArtifactRunnerProtocolError(
                "ArtifactProcessReceipt.binding must be an ArtifactRunnerBinding."
            )
        if not isinstance(self.result, ArtifactRunResult):
            raise ArtifactRunnerProtocolError(
                "ArtifactProcessReceipt.result must be an ArtifactRunResult."
            )
        if type(self.worker_process_started) is not bool:
            raise ArtifactRunnerProtocolError(
                "ArtifactProcessReceipt.worker_process_started must be boolean."
            )
        _require_int(
            self.schema_version,
            "ArtifactProcessReceipt.schema_version",
            minimum=ARTIFACT_RUNNER_SCHEMA_VERSION,
            maximum=ARTIFACT_RUNNER_SCHEMA_VERSION,
        )
        if (
            self.request.runner_binding_digest != self.binding.digest
            or self.request.checker_id is not self.binding.checker_id
            or self.request.checker_version != self.binding.checker_version
            or self.request.artifact_kind != self.binding.artifact_kind
            or self.request.artifact_manifest_digest
            != self.binding.artifact_manifest_digest
            or self.request.check_set_id != self.binding.check_set_id
            or self.request.check_set_manifest_digest
            != self.binding.check_set_manifest_digest
        ):
            raise ArtifactRunnerProtocolError(
                "Artifact run request does not match its runner binding."
            )
        if (
            self.result.checker_id is not self.request.checker_id
            or self.result.checker_version != self.request.checker_version
            or self.result.artifact_sha256 != self.request.artifact_sha256
        ):
            raise ArtifactRunnerProtocolError(
                "Artifact run result does not match its request."
            )
        codes = frozenset(self.result.outcome_codes)
        if self.result.outcome is ArtifactRunOutcome.COMPLETED:
            process_state_valid = self.worker_process_started
        elif self.result.outcome is ArtifactRunOutcome.INVALID_ARTIFACT:
            host_preflight = codes in (
                {ArtifactResultCode.EMPTY_INPUT},
                {ArtifactResultCode.INPUT_TOO_LARGE},
            )
            process_state_valid = (
                not self.worker_process_started
                if host_preflight
                else self.worker_process_started
            )
        elif self.result.outcome is ArtifactRunOutcome.TIMED_OUT:
            process_state_valid = self.worker_process_started
        elif self.result.outcome is ArtifactRunOutcome.WORKER_FAILED:
            process_state_valid = (
                not self.worker_process_started
                if codes == {ArtifactResultCode.WORKER_START_FAILED}
                else self.worker_process_started
            )
        else:
            process_state_valid = (
                not self.worker_process_started
                if codes == {ArtifactResultCode.BUNDLED_WORKER_MISMATCH}
                else self.worker_process_started
            )
        if not process_state_valid:
            raise ArtifactRunnerProtocolError(
                "Artifact process state is inconsistent with its result."
            )

    @property
    def digest(self) -> str:
        return canonical_digest(self.terms())

    def terms(self) -> dict[str, Any]:
        return {
            "artifact_executed": False,
            "binding": self.binding.terms(),
            "binding_digest": self.binding.digest,
            "certification_applied": False,
            "evaluation_created": False,
            "filesystem_isolation_enforced": False,
            "learner_projection_applied": False,
            "mastery_applied": False,
            "network_isolation_enforced": False,
            "operating_system_sandboxed": False,
            "process_separated": True,
            "request": self.request.terms(),
            "request_digest": self.request.digest,
            "result": self.result.terms(),
            "result_digest": self.result.digest,
            "schema_version": self.schema_version,
            "skill_authority": False,
            "synthetic": True,
            "trusted_checker_executed": self.worker_process_started,
            "worker_process_started": self.worker_process_started,
        }

    @classmethod
    def from_terms(cls, value: object) -> "ArtifactProcessReceipt":
        if type(value) is not dict:
            raise ArtifactRunnerProtocolError(
                "Artifact run receipt must be an object."
            )
        expected = frozenset(
            {
                "artifact_executed",
                "binding",
                "binding_digest",
                "certification_applied",
                "evaluation_created",
                "filesystem_isolation_enforced",
                "learner_projection_applied",
                "mastery_applied",
                "network_isolation_enforced",
                "operating_system_sandboxed",
                "process_separated",
                "request",
                "request_digest",
                "result",
                "result_digest",
                "schema_version",
                "skill_authority",
                "synthetic",
                "trusted_checker_executed",
                "worker_process_started",
            }
        )
        _require_exact_fields(value, expected, "Artifact run receipt")
        for field_name, expected_value in (
            ("artifact_executed", False),
            ("certification_applied", False),
            ("evaluation_created", False),
            ("filesystem_isolation_enforced", False),
            ("learner_projection_applied", False),
            ("mastery_applied", False),
            ("network_isolation_enforced", False),
            ("operating_system_sandboxed", False),
            ("process_separated", True),
            ("skill_authority", False),
            ("synthetic", True),
        ):
            _require_bool(value[field_name], expected_value, field_name)
        worker_started = value["worker_process_started"]
        if (
            type(worker_started) is not bool
            or value["trusted_checker_executed"] is not worker_started
        ):
            raise ArtifactRunnerProtocolError(
                "Artifact run receipt checker-process flags are inconsistent."
            )
        binding = ArtifactRunnerBinding.from_terms(value["binding"])
        request = ArtifactRunRequest.from_terms(value["request"])
        result = ArtifactRunResult.from_terms(value["result"])
        if (
            value["binding_digest"] != binding.digest
            or value["request_digest"] != request.digest
            or value["result_digest"] != result.digest
        ):
            raise ArtifactRunnerProtocolError(
                "Artifact run receipt contains a mismatched digest."
            )
        return cls(
            request=request,
            binding=binding,
            result=result,
            worker_process_started=worker_started,
            schema_version=value["schema_version"],
        )

    @classmethod
    def from_json(cls, raw: str | bytes) -> "ArtifactProcessReceipt":
        return cls.from_terms(
            _strict_json_object(
                raw,
                label="Artifact run receipt",
                maximum_bytes=128 * 1024,
            )
        )


def _worker_path() -> Path:
    return Path(__file__).with_name("_causal_mask_checker.py")


def _read_bundled_worker_source() -> bytes:
    try:
        source = _worker_path().read_bytes()
    except OSError as exc:
        raise ArtifactRunnerProtocolError(
            "Bundled artifact worker could not be read."
        ) from exc
    if not source or len(source) > MAX_BUNDLED_WORKER_BYTES:
        raise ArtifactRunnerProtocolError(
            "Bundled artifact worker has an invalid size."
        )
    return source


def bundled_synthetic_binding() -> ArtifactRunnerBinding:
    """Return the exact binding for the currently installed trusted worker."""

    source = _read_bundled_worker_source()
    if hashlib.sha256(source).hexdigest() != BUNDLED_WORKER_SHA256:
        raise ArtifactRunnerProtocolError(
            "Bundled artifact worker does not match its frozen v1 digest."
        )
    return ArtifactRunnerBinding(
        runner_id=SYNTHETIC_RUNNER_ID,
        runner_version=SYNTHETIC_RUNNER_VERSION,
        checker_id=ArtifactCheckerId.CAUSAL_MASK_MATRIX,
        checker_version="v1",
        artifact_kind=CAUSAL_MASK_ARTIFACT_KIND,
        artifact_manifest_digest=CAUSAL_MASK_ARTIFACT_MANIFEST_DIGEST,
        check_set_id=CAUSAL_MASK_CHECK_SET_ID,
        check_set_manifest_digest=CAUSAL_MASK_CHECK_SET_MANIFEST_DIGEST,
        worker_sha256=BUNDLED_WORKER_SHA256,
    )


def build_artifact_run_request(
    run_id: str,
    artifact_bytes: bytes,
    binding: ArtifactRunnerBinding,
) -> ArtifactRunRequest:
    """Bind one already captured byte snapshot to an exact synthetic checker."""

    if type(artifact_bytes) is not bytes:
        raise ArtifactRunnerProtocolError(
            "artifact_bytes must be an immutable bytes snapshot."
        )
    if not isinstance(binding, ArtifactRunnerBinding):
        raise ArtifactRunnerProtocolError(
            "binding must be an ArtifactRunnerBinding."
        )
    return ArtifactRunRequest(
        run_id=run_id,
        checker_id=binding.checker_id,
        checker_version=binding.checker_version,
        artifact_kind=binding.artifact_kind,
        artifact_manifest_digest=binding.artifact_manifest_digest,
        artifact_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
        artifact_size_bytes=len(artifact_bytes),
        check_set_id=binding.check_set_id,
        check_set_manifest_digest=binding.check_set_manifest_digest,
        runner_binding_digest=binding.digest,
    )


def _error_result(
    request: ArtifactRunRequest,
    outcome: ArtifactRunOutcome,
    code: ArtifactResultCode,
) -> ArtifactRunResult:
    return ArtifactRunResult(
        checker_id=request.checker_id,
        checker_version=request.checker_version,
        artifact_sha256=request.artifact_sha256,
        outcome=outcome,
        outcome_codes=(code,),
        passed=0,
        failed=0,
        errored=1,
        skipped=0,
    )


@dataclass(frozen=True, slots=True)
class _WorkerInvocation:
    result: ArtifactRunResult
    process_started: bool


def _write_all(descriptor: int, material: bytes) -> None:
    position = 0
    while position < len(material):
        written = os.write(descriptor, material[position:])
        if written <= 0:
            raise OSError("short write")
        position += written


def _bounded_pipe_reader(
    stream: BinaryIO,
    *,
    limit: int,
    shared_state: dict[str, Any],
    lock: threading.Lock,
    overflow: threading.Event,
    read_failed: threading.Event,
    process: subprocess.Popen[bytes],
    destination: bytearray,
) -> None:
    try:
        while True:
            chunk = stream.read(4_096)
            if not chunk:
                return
            with lock:
                remaining = max(0, limit - shared_state["total"])
                destination.extend(chunk[:remaining])
                shared_state["total"] += len(chunk)
                exceeded = shared_state["total"] > limit
            if exceeded:
                overflow.set()
                _terminate_process_group(process)
    except OSError:
        read_failed.set()


def _pipe_writer(
    stream: BinaryIO,
    material: bytes,
    write_failed: threading.Event,
) -> None:
    try:
        stream.write(material)
        stream.flush()
    except (BrokenPipeError, OSError):
        write_failed.set()
    finally:
        try:
            stream.close()
        except OSError:
            write_failed.set()


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    """Kill the private POSIX process group, falling back to the direct child."""

    if os.name == "posix" and process.pid > 0:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            return
        except (OSError, ProcessLookupError):
            pass
    try:
        process.kill()
    except OSError:
        pass


def _remaining_seconds(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _invoke_worker(
    request: ArtifactRunRequest,
    binding: ArtifactRunnerBinding,
    artifact_bytes: bytes,
    worker_source: bytes,
) -> _WorkerInvocation:
    """Invoke trusted worker source; exposed only for focused boundary tests."""

    if hashlib.sha256(worker_source).hexdigest() != binding.worker_sha256:
        return _WorkerInvocation(
            _error_result(
                request,
                ArtifactRunOutcome.PROTOCOL_ERROR,
                ArtifactResultCode.BUNDLED_WORKER_MISMATCH,
            ),
            False,
        )

    with tempfile.TemporaryDirectory(prefix="tsq-artifact-worker-") as raw:
        private_directory = Path(raw)
        script_path = private_directory / "worker.py"
        descriptor = -1
        close_failed = False
        try:
            descriptor = os.open(
                script_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o500,
            )
            _write_all(descriptor, worker_source)
            os.fsync(descriptor)
        except OSError:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                descriptor = -1
            return _WorkerInvocation(
                _error_result(
                    request,
                    ArtifactRunOutcome.WORKER_FAILED,
                    ArtifactResultCode.WORKER_START_FAILED,
                ),
                False,
            )
        finally:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    close_failed = True
        if close_failed:
            return _WorkerInvocation(
                _error_result(
                    request,
                    ArtifactRunOutcome.WORKER_FAILED,
                    ArtifactResultCode.WORKER_START_FAILED,
                ),
                False,
            )

        arguments = [
            sys.executable,
            "-I",
            "-S",
            str(script_path),
            "--protocol-version",
            str(ARTIFACT_WORKER_PROTOCOL_VERSION),
            "--checker-id",
            request.checker_id.value,
            "--checker-version",
            request.checker_version,
            "--artifact-sha256",
            request.artifact_sha256,
            "--maximum-input-bytes",
            str(binding.maximum_input_bytes),
        ]
        try:
            process = subprocess.Popen(
                arguments,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=private_directory,
                env={},
                close_fds=True,
                shell=False,
                start_new_session=True,
            )
        except OSError:
            return _WorkerInvocation(
                _error_result(
                    request,
                    ArtifactRunOutcome.WORKER_FAILED,
                    ArtifactResultCode.WORKER_START_FAILED,
                ),
                False,
            )

        stdout = bytearray()
        stderr = bytearray()
        overflow = threading.Event()
        read_failed = threading.Event()
        write_failed = threading.Event()
        shared_state: dict[str, Any] = {"total": 0}
        lock = threading.Lock()
        assert process.stdout is not None
        assert process.stderr is not None
        readers = (
            threading.Thread(
                target=_bounded_pipe_reader,
                kwargs={
                    "stream": process.stdout,
                    "limit": binding.maximum_output_bytes,
                    "shared_state": shared_state,
                    "lock": lock,
                    "overflow": overflow,
                    "read_failed": read_failed,
                    "process": process,
                    "destination": stdout,
                },
                daemon=True,
            ),
            threading.Thread(
                target=_bounded_pipe_reader,
                kwargs={
                    "stream": process.stderr,
                    "limit": binding.maximum_output_bytes,
                    "shared_state": shared_state,
                    "lock": lock,
                    "overflow": overflow,
                    "read_failed": read_failed,
                    "process": process,
                    "destination": stderr,
                },
                daemon=True,
            ),
        )
        assert process.stdin is not None
        writer = threading.Thread(
            target=_pipe_writer,
            args=(process.stdin, artifact_bytes, write_failed),
            daemon=True,
        )
        deadline = time.monotonic() + (binding.timeout_ms / 1_000)
        for reader in readers:
            reader.start()
        writer.start()

        timed_out = False
        return_code: int | None = None
        try:
            try:
                return_code = process.wait(_remaining_seconds(deadline))
            except subprocess.TimeoutExpired:
                timed_out = True
            if not timed_out:
                for thread in (writer, *readers):
                    thread.join(_remaining_seconds(deadline))
                    if thread.is_alive():
                        timed_out = True
                        break
            if timed_out:
                _terminate_process_group(process)
                try:
                    return_code = process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    return_code = process.poll()
        finally:
            if process.poll() is None:
                _terminate_process_group(process)
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            for thread in (writer, *readers):
                thread.join(timeout=1)
            if process.stdin is not None and not process.stdin.closed:
                try:
                    process.stdin.close()
                except OSError:
                    write_failed.set()
            if process.stdout is not None:
                try:
                    process.stdout.close()
                except OSError:
                    read_failed.set()
            if process.stderr is not None:
                try:
                    process.stderr.close()
                except OSError:
                    read_failed.set()

        if overflow.is_set():
            result = _error_result(
                request,
                ArtifactRunOutcome.PROTOCOL_ERROR,
                ArtifactResultCode.WORKER_OUTPUT_LIMIT,
            )
        elif timed_out:
            result = _error_result(
                request,
                ArtifactRunOutcome.TIMED_OUT,
                ArtifactResultCode.WORKER_TIMEOUT,
            )
        elif return_code != 0:
            result = _error_result(
                request,
                ArtifactRunOutcome.WORKER_FAILED,
                ArtifactResultCode.WORKER_EXIT_NONZERO,
            )
        elif write_failed.is_set() or read_failed.is_set():
            result = _error_result(
                request,
                ArtifactRunOutcome.PROTOCOL_ERROR,
                ArtifactResultCode.WORKER_PROTOCOL_INVALID,
            )
        elif stderr:
            result = _error_result(
                request,
                ArtifactRunOutcome.PROTOCOL_ERROR,
                ArtifactResultCode.WORKER_STDERR,
            )
        else:
            try:
                parsed = ArtifactRunResult.from_json(bytes(stdout))
                expected_wire = (
                    canonical_json(parsed.terms()) + "\n"
                ).encode("ascii")
                if (
                    bytes(stdout) != expected_wire
                    or parsed.checker_id is not request.checker_id
                    or parsed.checker_version != request.checker_version
                    or parsed.artifact_sha256 != request.artifact_sha256
                ):
                    raise ArtifactRunnerProtocolError(
                        "Worker result does not match its request."
                    )
                result = parsed
            except (ArtifactRunnerProtocolError, UnicodeError):
                result = _error_result(
                    request,
                    ArtifactRunOutcome.PROTOCOL_ERROR,
                    ArtifactResultCode.WORKER_PROTOCOL_INVALID,
                )
        return _WorkerInvocation(result, True)


class SyntheticArtifactRunnerRegistry:
    """Exact registry for bundled synthetic data checkers only.

    No application or learner callback can be registered.  Registration is an
    explicit test opt-in and the binding must byte-for-byte match the bundled
    worker and frozen manifests.
    """

    def __init__(self, *, allow_synthetic: bool = False) -> None:
        if type(allow_synthetic) is not bool:
            raise ArtifactRunnerProtocolError(
                "allow_synthetic must be boolean."
            )
        self._allow_synthetic = allow_synthetic
        self._bindings: dict[
            tuple[ArtifactCheckerId, str],
            ArtifactRunnerBinding,
        ] = {}

    def register(
        self,
        binding: ArtifactRunnerBinding,
    ) -> ArtifactRunnerBinding:
        if not self._allow_synthetic:
            raise ArtifactRunnerProtocolError(
                "Synthetic artifact checkers require allow_synthetic=True."
            )
        if not isinstance(binding, ArtifactRunnerBinding):
            raise ArtifactRunnerProtocolError(
                "Only ArtifactRunnerBinding values may be registered."
            )
        expected = bundled_synthetic_binding()
        if binding != expected:
            raise ArtifactRunnerProtocolError(
                "Only the exact bundled synthetic checker binding is allowed."
            )
        if binding.key in self._bindings:
            raise ArtifactRunnerProtocolError(
                "Artifact checker ID/version is already registered."
            )
        self._bindings[binding.key] = binding
        return binding

    def list(self) -> tuple[ArtifactRunnerBinding, ...]:
        return tuple(
            self._bindings[key]
            for key in sorted(
                self._bindings,
                key=lambda item: (item[0].value, item[1]),
            )
        )

    def inspect(
        self,
        checker_id: ArtifactCheckerId | str,
        checker_version: str,
    ) -> ArtifactRunnerBinding:
        try:
            typed_id = (
                checker_id
                if isinstance(checker_id, ArtifactCheckerId)
                else ArtifactCheckerId(checker_id)
            )
        except (TypeError, ValueError) as exc:
            raise ArtifactRunnerNotFoundError(
                "Artifact checker is not registered."
            ) from exc
        binding = self._bindings.get((typed_id, checker_version))
        if binding is None:
            raise ArtifactRunnerNotFoundError(
                "Artifact checker is not registered."
            )
        return binding

    def run(
        self,
        request: ArtifactRunRequest,
        artifact_bytes: bytes,
    ) -> ArtifactProcessReceipt:
        if not isinstance(request, ArtifactRunRequest):
            raise ArtifactRunnerProtocolError(
                "request must be an ArtifactRunRequest."
            )
        if type(artifact_bytes) is not bytes:
            raise ArtifactRunnerProtocolError(
                "artifact_bytes must be an immutable bytes snapshot."
            )
        binding = self.inspect(request.checker_id, request.checker_version)
        if (
            request.runner_binding_digest != binding.digest
            or request.artifact_kind != binding.artifact_kind
            or request.artifact_manifest_digest
            != binding.artifact_manifest_digest
            or request.check_set_id != binding.check_set_id
            or request.check_set_manifest_digest
            != binding.check_set_manifest_digest
        ):
            raise ArtifactRunnerProtocolError(
                "Artifact run request does not match its registered binding."
            )
        if request.artifact_size_bytes != len(artifact_bytes):
            raise ArtifactRunnerProtocolError(
                "Artifact byte length does not match the run request."
            )
        if (
            hashlib.sha256(artifact_bytes).hexdigest()
            != request.artifact_sha256
        ):
            raise ArtifactRunnerProtocolError(
                "Artifact bytes do not match the run request digest."
            )

        if not artifact_bytes:
            invocation = _WorkerInvocation(
                _error_result(
                    request,
                    ArtifactRunOutcome.INVALID_ARTIFACT,
                    ArtifactResultCode.EMPTY_INPUT,
                ),
                False,
            )
        elif len(artifact_bytes) > binding.maximum_input_bytes:
            invocation = _WorkerInvocation(
                _error_result(
                    request,
                    ArtifactRunOutcome.INVALID_ARTIFACT,
                    ArtifactResultCode.INPUT_TOO_LARGE,
                ),
                False,
            )
        else:
            source = _read_bundled_worker_source()
            if hashlib.sha256(source).hexdigest() != binding.worker_sha256:
                invocation = _WorkerInvocation(
                    _error_result(
                        request,
                        ArtifactRunOutcome.PROTOCOL_ERROR,
                        ArtifactResultCode.BUNDLED_WORKER_MISMATCH,
                    ),
                    False,
                )
            else:
                invocation = _invoke_worker(
                    request,
                    binding,
                    artifact_bytes,
                    source,
                )
        return ArtifactProcessReceipt(
            request=request,
            binding=binding,
            result=invocation.result,
            worker_process_started=invocation.process_started,
        )

__all__ = [
    "ARTIFACT_RUNNER_SCHEMA_VERSION",
    "ARTIFACT_WORKER_PROTOCOL_VERSION",
    "BUNDLED_WORKER_SHA256",
    "CAUSAL_MASK_ARTIFACT_KIND",
    "CAUSAL_MASK_ARTIFACT_MANIFEST_DIGEST",
    "CAUSAL_MASK_CHECK_SET_ID",
    "CAUSAL_MASK_CHECK_SET_MANIFEST_DIGEST",
    "DEFAULT_WORKER_TIMEOUT_MS",
    "MAX_RUNNER_ARTIFACT_BYTES",
    "MAX_WORKER_OUTPUT_BYTES",
    "ArtifactCheckerId",
    "ArtifactProcessReceipt",
    "ArtifactResultCode",
    "ArtifactRunOutcome",
    "ArtifactRunRequest",
    "ArtifactRunResult",
    "ArtifactRunnerBinding",
    "ArtifactRunnerNotFoundError",
    "ArtifactRunnerProtocolError",
    "SyntheticArtifactRunnerRegistry",
    "build_artifact_run_request",
    "bundled_synthetic_binding",
]
