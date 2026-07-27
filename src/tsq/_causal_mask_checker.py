# SPDX-License-Identifier: MPL-2.0

"""Standalone worker for TSQ's inert causal-mask-matrix fixture.

This file is intentionally self-contained so the host can copy it into a
private temporary directory and start it with ``python -I -S``.  It parses a
small JSON data document.  It never imports, compiles, evaluates, or executes
learner-provided bytes.

Process separation is defense in depth, not an operating-system sandbox.  The
worker is trusted application code and Python still has the ambient authority
of its operating-system account.  Safety therefore depends on the deliberately
closed data-only checker below; arbitrary learner code is not supported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from typing import Any


PROTOCOL_VERSION = 1
CHECKER_ID = "synthetic.causal-mask-matrix"
CHECKER_VERSION = "v1"
MAX_JSON_DEPTH = 8
MAX_MATRIX_SIZE = 64
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class _DuplicateFieldError(ValueError):
    pass


class _NonFiniteNumberError(ValueError):
    pass


class _UnexpectedNumberError(ValueError):
    pass


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateFieldError
        result[key] = value
    return result


def _reject_constant(_raw: str) -> None:
    raise _NonFiniteNumberError


def _parse_integer(raw: str) -> int:
    # The only accepted integer is the small schema version.  Bounding this
    # parser also avoids allocating an arbitrarily large Python integer.
    if len(raw) > 16:
        raise _UnexpectedNumberError
    return int(raw)


def _reject_float(_raw: str) -> float:
    raise _UnexpectedNumberError


def _maximum_json_depth(text: str) -> int:
    """Measure container nesting without interpreting string contents."""

    depth = 0
    maximum = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            maximum = max(maximum, depth)
        elif character in "]}":
            depth -= 1
    return maximum


def _result(
    artifact_sha256: str,
    outcome: str,
    outcome_codes: tuple[str, ...],
    *,
    passed: int = 0,
    failed: int = 0,
    errored: int = 0,
    skipped: int = 0,
) -> dict[str, Any]:
    return {
        "artifact_sha256": artifact_sha256,
        "checker_id": CHECKER_ID,
        "checker_version": CHECKER_VERSION,
        "errored": errored,
        "failed": failed,
        "outcome": outcome,
        "outcome_codes": sorted(outcome_codes),
        "passed": passed,
        "schema_version": PROTOCOL_VERSION,
        "skipped": skipped,
    }


def _invalid(artifact_sha256: str, code: str) -> dict[str, Any]:
    return _result(
        artifact_sha256,
        "invalid_artifact",
        (code,),
        errored=1,
    )


def _check(
    raw: bytes,
    *,
    expected_digest: str,
    maximum_input_bytes: int,
) -> dict[str, Any]:
    if not raw:
        return _invalid(expected_digest, "empty_input")
    if len(raw) > maximum_input_bytes:
        return _invalid(expected_digest, "input_too_large")

    observed_digest = hashlib.sha256(raw).hexdigest()
    if observed_digest != expected_digest:
        return _invalid(observed_digest, "artifact_digest_mismatch")

    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return _invalid(observed_digest, "invalid_utf8")

    if _maximum_json_depth(text) > MAX_JSON_DEPTH:
        return _invalid(observed_digest, "json_depth_exceeded")

    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
            parse_int=_parse_integer,
            parse_float=_reject_float,
        )
    except _DuplicateFieldError:
        return _invalid(observed_digest, "duplicate_field")
    except _NonFiniteNumberError:
        return _invalid(observed_digest, "nonfinite_number")
    except (_UnexpectedNumberError, json.JSONDecodeError, RecursionError):
        return _invalid(observed_digest, "invalid_json")

    if type(value) is not dict:
        return _invalid(observed_digest, "invalid_document")
    if set(value) != {"mask", "schema_version"}:
        return _invalid(observed_digest, "unknown_or_missing_fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        return _invalid(observed_digest, "invalid_schema_version")

    matrix = value["mask"]
    if (
        type(matrix) is not list
        or not 1 <= len(matrix) <= MAX_MATRIX_SIZE
    ):
        return _invalid(observed_digest, "invalid_matrix")
    size = len(matrix)
    if any(
        type(row) is not list
        or len(row) != size
        or any(type(cell) is not bool for cell in row)
        for row in matrix
    ):
        return _invalid(observed_digest, "invalid_matrix")

    causal = all(
        matrix[row][column] is (column <= row)
        for row in range(size)
        for column in range(size)
    )
    if causal:
        return _result(
            observed_digest,
            "completed",
            ("matrix_shape_valid", "causal_visibility_valid"),
            passed=2,
        )
    return _result(
        observed_digest,
        "completed",
        ("matrix_shape_valid", "causal_visibility_invalid"),
        passed=1,
        failed=1,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--protocol-version", type=int, required=True)
    parser.add_argument("--checker-id", required=True)
    parser.add_argument("--checker-version", required=True)
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("--maximum-input-bytes", type=int, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if (
        arguments.protocol_version != PROTOCOL_VERSION
        or arguments.checker_id != CHECKER_ID
        or arguments.checker_version != CHECKER_VERSION
        or not _DIGEST_PATTERN.fullmatch(arguments.artifact_sha256)
        or not 1 <= arguments.maximum_input_bytes <= 65_536
    ):
        return 2

    raw = sys.stdin.buffer.read(arguments.maximum_input_bytes + 1)
    result = _check(
        raw,
        expected_digest=arguments.artifact_sha256,
        maximum_input_bytes=arguments.maximum_input_bytes,
    )
    encoded = json.dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    sys.stdout.buffer.write(encoded + b"\n")
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
