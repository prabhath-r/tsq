# SPDX-License-Identifier: MPL-2.0

class TSQError(Exception):
    """Base class for errors that are safe to surface to a CLI caller."""


class ValidationError(TSQError):
    """A domain object or corpus bundle violated an invariant."""

    def __init__(self, message: str, *, issues=()):
        super().__init__(message)
        self.issues = tuple(issues)


class NotFoundError(TSQError):
    """A requested learner, session, concept, or item does not exist."""


class ConflictError(TSQError):
    """A command conflicts with current state or repeats inconsistently."""


class ExhaustedError(TSQError):
    """No eligible item remains under the current policy constraints."""
