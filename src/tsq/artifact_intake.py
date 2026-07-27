# SPDX-License-Identifier: MPL-2.0

"""Safe intake for learner-created productive artifacts.

The digest-only checkpoint path opens an explicitly named local file only long
enough to compute its SHA-256 digest.  It does not parse, import, execute,
retain, copy, or name the file in its result.  The explicit snapshot path
retains one verified immutable byte copy in memory for a trusted inert checker;
it never carries the source path or persists the content.  Neither path is an
evaluation or a skill claim.  A digest is a commitment, not encryption:
callers should still protect guessable or sensitive source material outside
TSQ.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

from .evidence import ActionKind
from .errors import ValidationError


MAX_PRODUCTIVE_ARTIFACT_BYTES = 16 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CheckpointFileKind(StrEnum):
    """Closed learner-artifact checkpoints supported by file intake."""

    ARTIFACT = "artifact"
    EXPLANATION = "explanation"
    SUBMISSION = "submission"


_ACTION_BY_KIND: Mapping[CheckpointFileKind, ActionKind] = MappingProxyType(
    {
        CheckpointFileKind.ARTIFACT: ActionKind.ARTIFACT_CHECKPOINT,
        CheckpointFileKind.EXPLANATION: ActionKind.EXPLANATION_CHECKPOINT,
        CheckpointFileKind.SUBMISSION: ActionKind.SUBMITTED,
    }
)


def _file_fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    """Return the immutable identity terms checked across the complete read."""

    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _validate_regular_size(value: os.stat_result) -> None:
    if stat.S_ISLNK(value.st_mode):
        raise ValidationError("Productive artifact symlinks are not allowed.")
    if not stat.S_ISREG(value.st_mode):
        raise ValidationError("Productive artifact must be a regular file.")
    if value.st_size <= 0:
        raise ValidationError("Productive artifact must not be empty.")
    if value.st_size > MAX_PRODUCTIVE_ARTIFACT_BYTES:
        raise ValidationError(
            "Productive artifact exceeds the 16 MiB intake limit."
        )


def _read_productive_artifact(
    path: os.PathLike[str] | str,
    *,
    retain_content: bool,
) -> tuple[str, int, bytes | None]:
    """Read one stable regular file through the shared intake boundary.

    The path is inspected before opening, the descriptor is opened with
    ``O_NOFOLLOW`` where the platform provides it, and both descriptor and path
    identity are checked again after the streaming read.  Any replacement,
    resizing, timestamp change, short read, or unsafe file type fails closed.
    The final path component must not be a symlink; parent resolution follows
    normal operating-system path semantics. Error messages deliberately omit
    the caller's path.
    """

    try:
        path_value = os.fspath(path)
    except TypeError as exc:
        raise ValidationError(
            "Productive artifact path must identify a local file."
        ) from exc
    if not isinstance(path_value, (str, bytes)):
        raise ValidationError(
            "Productive artifact path must identify a local file."
        )

    try:
        before = os.lstat(path_value)
    except (OSError, ValueError) as exc:
        raise ValidationError(
            "Productive artifact could not be inspected."
        ) from exc
    _validate_regular_size(before)

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    # A regular file ignores O_NONBLOCK.  Its presence prevents a pathname
    # replacement with a FIFO from blocking between lstat and fstat.
    flags |= getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_BINARY", 0)

    try:
        descriptor = os.open(path_value, flags)
    except (OSError, ValueError) as exc:
        raise ValidationError(
            "Productive artifact could not be opened safely."
        ) from exc

    digest = hashlib.sha256()
    byte_count = 0
    content_parts: list[bytes] | None = [] if retain_content else None
    failure_in_flight = False
    try:
        try:
            opened = os.fstat(descriptor)
        except OSError as exc:
            raise ValidationError(
                "Productive artifact descriptor could not be inspected."
            ) from exc
        _validate_regular_size(opened)
        if _file_fingerprint(opened) != _file_fingerprint(before):
            raise ValidationError(
                "Productive artifact changed before it could be read safely."
            )

        try:
            while True:
                chunk = os.read(descriptor, _READ_CHUNK_BYTES)
                if not chunk:
                    break
                byte_count += len(chunk)
                if (
                    byte_count > opened.st_size
                    or byte_count > MAX_PRODUCTIVE_ARTIFACT_BYTES
                ):
                    raise ValidationError(
                        "Productive artifact changed while it was being read."
                    )
                digest.update(chunk)
                if content_parts is not None:
                    content_parts.append(chunk)
        except ValidationError:
            raise
        except OSError as exc:
            raise ValidationError(
                "Productive artifact could not be read safely."
            ) from exc

        try:
            descriptor_after = os.fstat(descriptor)
            path_after = os.lstat(path_value)
        except (OSError, ValueError) as exc:
            raise ValidationError(
                "Productive artifact changed while it was being read."
            ) from exc
        fingerprints = {
            _file_fingerprint(before),
            _file_fingerprint(opened),
            _file_fingerprint(descriptor_after),
            _file_fingerprint(path_after),
        }
        if len(fingerprints) != 1 or byte_count != opened.st_size:
            raise ValidationError(
                "Productive artifact changed while it was being read."
            )
    except BaseException:
        failure_in_flight = True
        raise
    finally:
        try:
            os.close(descriptor)
        except OSError as exc:
            if not failure_in_flight:
                raise ValidationError(
                    "Productive artifact descriptor could not be closed safely."
                ) from exc

    content = b"".join(content_parts) if content_parts is not None else None
    return digest.hexdigest(), byte_count, content


@dataclass(frozen=True, slots=True)
class ProductiveArtifactSnapshot:
    """One path-free, immutable in-memory artifact snapshot."""

    content: bytes = field(repr=False)
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if type(self.content) is not bytes:
            raise ValidationError(
                "Productive artifact snapshot content must be immutable bytes."
            )
        if (
            type(self.size_bytes) is not int
            or self.size_bytes <= 0
            or self.size_bytes > MAX_PRODUCTIVE_ARTIFACT_BYTES
            or self.size_bytes != len(self.content)
        ):
            raise ValidationError(
                "Productive artifact snapshot size does not match its content."
            )
        if (
            type(self.sha256) is not str
            or not _DIGEST_PATTERN.fullmatch(self.sha256)
            or hashlib.sha256(self.content).hexdigest() != self.sha256
        ):
            raise ValidationError(
                "Productive artifact snapshot digest does not match its content."
            )


def capture_productive_artifact(
    path: os.PathLike[str] | str,
) -> ProductiveArtifactSnapshot:
    """Capture one stable regular file without interpreting or naming it."""

    sha256, size_bytes, content = _read_productive_artifact(
        path,
        retain_content=True,
    )
    if content is None:  # pragma: no cover - fixed by the retain_content call
        raise AssertionError("Artifact intake failed to retain content.")
    return ProductiveArtifactSnapshot(
        content=content,
        sha256=sha256,
        size_bytes=size_bytes,
    )


def hash_productive_artifact(path: os.PathLike[str] | str) -> str:
    """Hash one stable regular file without interpreting or retaining its bytes."""

    sha256, _, _ = _read_productive_artifact(path, retain_content=False)
    return sha256


@dataclass(frozen=True, slots=True)
class PreparedFileCheckpoint:
    """Digest-only action material derived from one safely read local file."""

    kind: CheckpointFileKind
    action_kind: ActionKind
    sha256: str
    artifact_kind: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CheckpointFileKind):
            raise ValidationError(
                "Prepared checkpoint kind must be a CheckpointFileKind."
            )
        if self.action_kind is not _ACTION_BY_KIND[self.kind]:
            raise ValidationError(
                "Prepared checkpoint action does not match its file kind."
            )
        if not isinstance(self.sha256, str) or not _DIGEST_PATTERN.fullmatch(
            self.sha256
        ):
            raise ValidationError(
                "Prepared checkpoint digest must be lowercase SHA-256."
            )
        if self.kind is CheckpointFileKind.ARTIFACT:
            if (
                not isinstance(self.artifact_kind, str)
                or not _ID_PATTERN.fullmatch(self.artifact_kind)
            ):
                raise ValidationError(
                    "Prepared artifact checkpoint requires a stable artifact kind."
                )
        elif self.artifact_kind is not None:
            raise ValidationError(
                "Only prepared artifact checkpoints may carry an artifact kind."
            )

    @property
    def payload(self) -> Mapping[str, Any]:
        if self.kind is CheckpointFileKind.ARTIFACT:
            return MappingProxyType(
                {
                    "artifact_digest": self.sha256,
                    "artifact_kind": self.artifact_kind,
                }
            )
        if self.kind is CheckpointFileKind.EXPLANATION:
            return MappingProxyType({"explanation_digest": self.sha256})
        return MappingProxyType({"submission_digest": self.sha256})


def prepare_file_checkpoint(
    path: os.PathLike[str] | str,
    *,
    kind: str,
    artifact_kind: str | None = None,
) -> PreparedFileCheckpoint:
    """Validate a closed checkpoint request, then hash its file safely."""

    try:
        typed_kind = CheckpointFileKind(kind)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "Checkpoint file kind must be artifact, explanation, or submission."
        ) from exc
    if typed_kind is CheckpointFileKind.ARTIFACT:
        if artifact_kind is None:
            raise ValidationError(
                "Artifact checkpoints require --artifact-kind."
            )
        if not isinstance(artifact_kind, str) or not _ID_PATTERN.fullmatch(
            artifact_kind
        ):
            raise ValidationError(
                "Artifact kind must be a stable identifier."
            )
    elif artifact_kind is not None:
        raise ValidationError(
            "--artifact-kind is allowed only for artifact checkpoints."
        )

    sha256 = hash_productive_artifact(path)
    return PreparedFileCheckpoint(
        kind=typed_kind,
        action_kind=_ACTION_BY_KIND[typed_kind],
        sha256=sha256,
        artifact_kind=artifact_kind,
    )


__all__ = [
    "CheckpointFileKind",
    "MAX_PRODUCTIVE_ARTIFACT_BYTES",
    "PreparedFileCheckpoint",
    "ProductiveArtifactSnapshot",
    "capture_productive_artifact",
    "hash_productive_artifact",
    "prepare_file_checkpoint",
]
