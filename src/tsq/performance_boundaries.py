# SPDX-License-Identifier: MPL-2.0

"""Shared semantic boundaries for released productive-skill tasks."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping

from .evidence import LearningTask


def release_misconception_objectives(
    connection: sqlite3.Connection,
    release_id: str,
    *,
    accepted_only: bool,
    exclude_revoked: bool,
) -> dict[str, set[str]]:
    """Return option-level objective bindings under an explicit live boundary."""

    clauses = [
        "mapping.release_id = ?",
        "option.is_correct = 0",
        "option.misconception_id IS NOT NULL",
    ]
    if accepted_only:
        clauses.append("membership.status IN ('approved', 'calibrated')")
    if exclude_revoked:
        clauses.append(
            """NOT EXISTS (
                   SELECT 1
                   FROM question_revocations revoked
                   WHERE revoked.question_id = mapping.question_id
               )"""
        )
    mappings: dict[str, set[str]] = {}
    for row in connection.execute(
        """SELECT DISTINCT option.misconception_id,
                          mapping.objective_id
           FROM release_option_objectives mapping
           JOIN release_questions membership
             ON membership.release_id = mapping.release_id
            AND membership.question_id = mapping.question_id
           JOIN options option
             ON option.question_id = mapping.question_id
            AND option.option_id = mapping.option_id
           WHERE """
        + " AND ".join(clauses)
        + " ORDER BY option.misconception_id, mapping.objective_id",
        (release_id,),
    ):
        mappings.setdefault(row["misconception_id"], set()).add(
            row["objective_id"]
        )
    return mappings


def missing_objective_misconception_bindings(
    task: LearningTask,
    misconception_objectives: Mapping[str, set[str]],
) -> tuple[tuple[str, str], ...]:
    """Return criterion/misconception pairs lacking an objective intersection."""

    missing: list[tuple[str, str]] = []
    for criterion in task.criteria:
        objective_ids = set(criterion.objective_ids)
        if not objective_ids:
            continue
        for misconception_id in criterion.misconception_ids:
            if not (
                objective_ids
                & misconception_objectives.get(misconception_id, set())
            ):
                missing.append((criterion.id, misconception_id))
    return tuple(missing)
