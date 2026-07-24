# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import hashlib
import json
import math
import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import median
from types import MappingProxyType
from typing import Iterable, Mapping

from .adaptive import RecursiveEvidenceBoundary
from .capacity import VERIFICATION_KINDS
from .errors import ConflictError, ExhaustedError, ValidationError
from .inference import classify_response_for_model
from .learner import (
    OBJECTIVE_MODEL_VERSIONS,
    SPACING_AWARE_FAMILY_MODEL_VERSIONS,
    LearnerModel,
)
from .models import (
    MAX_REMEDIATION_DEPTH,
    CandidateScore,
    ObjectiveState,
    Presentation,
    Question,
    SessionPhase,
)
from .store import Database, new_id, question_runtime_activation_safe
from .versions import (
    SUPPORTED_MODEL_VERSIONS,
    question_selected_schema_for,
)


PERSISTENT_GAP_EPISODE_POLICY_VERSION = "recursive-evidence-graph-v13"
ACTIVE_MISCONCEPTION_REVISIT_POLICY_VERSION = "recursive-evidence-graph-v14"
EXACT_OBJECTIVE_READINESS_POLICY_VERSION = "recursive-evidence-graph-v15"
EXPLORATION_FALLBACK_POLICY_VERSION = "recursive-evidence-graph-v16"
HYBRID_COVERAGE_POLICY_VERSION = "recursive-evidence-graph-v17"
POLICY_VERSION = HYBRID_COVERAGE_POLICY_VERSION
PERSISTENT_GAP_EPISODE_POLICY_VERSIONS = frozenset(
    {
        PERSISTENT_GAP_EPISODE_POLICY_VERSION,
        ACTIVE_MISCONCEPTION_REVISIT_POLICY_VERSION,
        EXACT_OBJECTIVE_READINESS_POLICY_VERSION,
        EXPLORATION_FALLBACK_POLICY_VERSION,
        HYBRID_COVERAGE_POLICY_VERSION,
    }
)
PERSISTENT_GAP_MIN_OBSERVED_FAMILIES = 2
PERSISTENT_GAP_COMPARISON_EPSILON = 1e-12
PERSISTENT_GAP_EPISODE_BUDGET = 2
ACTIVE_MISCONCEPTION_REVISIT_THRESHOLD = 0.35
HYBRID_COVERAGE_RAW_BURDEN_SLACK = 1
CANDIDATE_SAMPLING_FRONTIER_LIMIT = 5
CANDIDATE_AUDIT_PREFIX_LIMIT = 10
_PERSISTENT_GAP_BASE_MARKERS = frozenset(
    {
        "persistent_gap_revisit",
        "persistent_gap_observed_families",
        "persistent_gap_mastery",
        "persistent_gap_cold_prior",
        "persistent_gap_due_at",
    }
)
_PERSISTENT_GAP_EPISODE_MARKERS = frozenset(
    {
        "persistent_gap_episode_spend",
        "persistent_gap_episode_budget",
    }
)


class _RetrySelection(Exception):
    pass


class _ExplorationUnserviceable(Exception):
    """Ask the caller to retry the unchanged session in its requested scope."""

    def __init__(self, reason: str, topic_ids: tuple[str, ...]):
        super().__init__(reason)
        self.reason = reason
        self.topic_ids = topic_ids


@dataclass(frozen=True, slots=True)
class _HybridCoverage:
    """Three non-interchangeable views of one adaptive target.

    ``raw_exposures`` limits repeated learner burden. ``diagnostic_information``
    is the target projection's cumulative quality- and dependence-discounted
    evidence mass, so credible errors count while abstention, hints, low
    confidence, and implausible latency contribute only their learner-model
    discount. ``successful_retrieval_families`` remains a separate positive
    competence signal and is never inferred from the other two values.
    """

    raw_exposures: int
    diagnostic_information: float
    successful_retrieval_families: int

    def __post_init__(self) -> None:
        if type(self.raw_exposures) is not int or self.raw_exposures < 0:
            raise ValidationError(
                "Hybrid coverage raw exposures must be a non-negative integer."
            )
        if (
            isinstance(self.diagnostic_information, bool)
            or not isinstance(self.diagnostic_information, (int, float))
            or not math.isfinite(float(self.diagnostic_information))
            or self.diagnostic_information < 0.0
        ):
            raise ValidationError(
                "Hybrid coverage diagnostic information must be finite and "
                "non-negative."
            )
        if (
            type(self.successful_retrieval_families) is not int
            or self.successful_retrieval_families < 0
            or self.successful_retrieval_families > self.raw_exposures
        ):
            raise ValidationError(
                "Hybrid coverage successful retrieval families must be a "
                "non-negative integer no greater than raw exposures."
            )


def _selection_version_boundary(
    connection: sqlite3.Connection,
    *,
    decision_id: str,
    session_id: str,
) -> tuple[str, str, datetime]:
    """Return and validate the immutable selection boundary.

    Decisions deliberately do not duplicate this event metadata.  A pending
    promise is safe to reuse only when it has one unambiguous selection anchor.
    """
    rows = connection.execute(
        """SELECT selection.schema_version, selection.occurred_at,
                  selection.metadata_json, decision.created_at,
                  decision.policy_version, decision.corpus_release_id,
                  decision.question_objective_id,
                  decision.focus_objective_id
           FROM events selection
           JOIN decisions decision ON decision.id = ?
           WHERE selection.event_type = 'QuestionSelected'
             AND selection.session_id = ?
             AND json_extract(
                 selection.payload_json, '$.decision_id'
             ) = decision.id
           ORDER BY selection.stream_version""",
        (decision_id, session_id),
    ).fetchall()
    if len(rows) != 1:
        raise ValidationError(
            f"Pending decision {decision_id} lacks a unique QuestionSelected "
            "model boundary."
        )
    try:
        metadata = json.loads(rows[0]["metadata_json"])
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"Pending decision {decision_id} has invalid QuestionSelected metadata."
        ) from exc
    if type(metadata) is not dict or set(metadata) != {
        "policy_version",
        "learner_model_version",
        "corpus_release_id",
    }:
        raise ValidationError(
            f"Pending decision {decision_id} has invalid QuestionSelected metadata."
        )
    model_version = (
        metadata.get("learner_model_version")
    )
    if (
        not isinstance(model_version, str)
        or not model_version
        or model_version not in SUPPORTED_MODEL_VERSIONS
    ):
        raise ValidationError(
            f"Pending decision {decision_id} has no supported selection learner "
            "model."
        )
    policy_version = metadata.get("policy_version")
    if not isinstance(policy_version, str) or not policy_version:
        raise ValidationError(
            f"Pending decision {decision_id} has no valid selection policy."
        )
    row = rows[0]
    if (
        policy_version != row["policy_version"]
        or metadata.get("corpus_release_id") != row["corpus_release_id"]
    ):
        raise ValidationError(
            f"Pending decision {decision_id} QuestionSelected metadata does "
            "not match its decision."
        )
    objective_aware = bool(
        row["question_objective_id"] is not None
        or row["focus_objective_id"] is not None
    )
    expected_schema = question_selected_schema_for(
        model_version, objective_aware=objective_aware
    )
    if row["schema_version"] != expected_schema:
        raise ValidationError(
            f"Pending decision {decision_id} QuestionSelected schema does not "
            f"match learner model {model_version}."
        )
    try:
        selected_at = datetime.fromisoformat(row["occurred_at"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(
            f"Pending decision {decision_id} has an invalid selection time."
        ) from exc
    if selected_at.tzinfo is None or selected_at.utcoffset() is None:
        raise ValidationError(
            f"Pending decision {decision_id} has a timezone-naive selection time."
        )
    if (
        expected_schema == 3
        and row["occurred_at"] != row["created_at"]
    ):
        raise ValidationError(
            f"Pending decision {decision_id} selection time is not bound to "
            "its decision."
        )
    return model_version, policy_version, selected_at


def _selection_learner_model_version(
    connection: sqlite3.Connection,
    *,
    decision_id: str,
    session_id: str,
) -> str:
    """Compatibility wrapper for older internal callers."""

    return _selection_version_boundary(
        connection,
        decision_id=decision_id,
        session_id=session_id,
    )[0]


_PHASE_WEIGHTS: dict[SessionPhase, dict[str, float]] = {
    SessionPhase.DIAGNOSE: {
        "information_gain": 0.28,
        "learning_fit": 0.08,
        "concept_need": 0.18,
        "misconception_value": 0.14,
        "prerequisite_value": 0.10,
        "review_value": 0.03,
        "novelty": 0.08,
        "kind_fit": 0.03,
        "boundary_fit": 0.08,
    },
    SessionPhase.LEARN: {
        "information_gain": 0.15,
        "learning_fit": 0.22,
        "concept_need": 0.19,
        "misconception_value": 0.12,
        "prerequisite_value": 0.06,
        "review_value": 0.05,
        "novelty": 0.08,
        "kind_fit": 0.03,
        "boundary_fit": 0.10,
    },
    SessionPhase.REMEDIATE: {
        "information_gain": 0.16,
        "learning_fit": 0.16,
        "concept_need": 0.15,
        "misconception_value": 0.27,
        "prerequisite_value": 0.06,
        "review_value": 0.00,
        "novelty": 0.08,
        "kind_fit": 0.05,
        "boundary_fit": 0.07,
    },
    SessionPhase.VERIFY: {
        "information_gain": 0.12,
        "learning_fit": 0.14,
        "concept_need": 0.14,
        "misconception_value": 0.12,
        "prerequisite_value": 0.05,
        "review_value": 0.00,
        "novelty": 0.13,
        "kind_fit": 0.25,
        "boundary_fit": 0.05,
    },
    SessionPhase.REVIEW: {
        "information_gain": 0.12,
        "learning_fit": 0.16,
        "concept_need": 0.10,
        "misconception_value": 0.08,
        "prerequisite_value": 0.02,
        "review_value": 0.32,
        "novelty": 0.09,
        "kind_fit": 0.06,
        "boundary_fit": 0.05,
    },
}


_TARGET_SUCCESS = {
    SessionPhase.DIAGNOSE: 0.52,
    SessionPhase.LEARN: 0.68,
    SessionPhase.REMEDIATE: 0.72,
    SessionPhase.VERIFY: 0.64,
    SessionPhase.REVIEW: 0.74,
}


_KIND_FIT = {
    SessionPhase.DIAGNOSE: {
        "diagnostic": 1.0,
        "prerequisite_probe": 0.95,
        "comparison": 0.75,
    },
    SessionPhase.LEARN: {
        "conceptual": 0.90,
        "application": 1.0,
        "calculation": 0.85,
        "debugging": 0.85,
    },
    SessionPhase.REMEDIATE: {
        "prerequisite_probe": 1.0,
        "debugging": 0.95,
        "counterfactual": 0.90,
        "comparison": 0.90,
    },
    SessionPhase.VERIFY: {
        "transfer": 1.0,
        "application": 0.95,
        "counterfactual": 0.95,
        "debugging": 0.85,
    },
    SessionPhase.REVIEW: {
        "transfer": 1.0,
        "application": 0.95,
        "conceptual": 0.75,
    },
}


class AdaptivePolicy:
    """Candidate retrieval, constrained scoring, and randomized top-k choice."""

    def __init__(self, database: Database, learner_model: LearnerModel | None = None):
        self.database = database
        self.learner_model = learner_model or LearnerModel()
        self.boundary_planner = RecursiveEvidenceBoundary(self.learner_model)

    @staticmethod
    def _is_due_persistent_gap(
        projected: ObjectiveState,
        cold_start: ObjectiveState,
        *,
        observed_response_families: int,
        now: datetime,
    ) -> bool:
        """Recognize a durable evidence gap without mistaking counts for skill."""

        if not AdaptivePolicy._has_persistent_gap(
            projected,
            cold_start,
            observed_response_families=observed_response_families,
        ):
            return False
        due_at = projected.next_review_at
        if due_at is None:
            return False
        if (
            now.tzinfo is None
            or now.utcoffset() is None
            or due_at.tzinfo is None
            or due_at.utcoffset() is None
        ):
            raise ValidationError(
                "Persistent-gap review timestamps must be timezone-aware."
            )
        return due_at.astimezone(timezone.utc) <= now.astimezone(timezone.utc)

    @staticmethod
    def _has_persistent_gap(
        projected: ObjectiveState,
        cold_start: ObjectiveState,
        *,
        observed_response_families: int,
    ) -> bool:
        """Whether exact mastery remains meaningfully below its cold prior."""

        if (
            type(observed_response_families) is not int
            or observed_response_families < 0
        ):
            raise ValidationError(
                "Observed objective-family count must be a non-negative integer."
            )
        if observed_response_families < PERSISTENT_GAP_MIN_OBSERVED_FAMILIES:
            return False
        if projected.objective_id != cold_start.objective_id:
            raise ValidationError(
                "Persistent-gap comparison crossed learning objectives."
            )

        current_mastery = projected.mastery_probability
        cold_mastery = cold_start.mastery_probability
        numerical_error = (
            projected.mastery_probability_error_bound
            + cold_start.mastery_probability_error_bound
            + PERSISTENT_GAP_COMPARISON_EPSILON
        )
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in (current_mastery, cold_mastery, numerical_error)
        ):
            raise ValidationError(
                "Persistent-gap mastery comparison must be finite and non-negative."
            )
        return cold_mastery - current_mastery > numerical_error

    def _observed_objective_response_families(
        self,
        learner_id: str,
        objective_ids: set[str],
    ) -> dict[str, int]:
        """Count distinct attempted families, not repeated raw responses."""

        counts = {objective_id: 0 for objective_id in objective_ids}
        if not objective_ids:
            return counts
        placeholders = ",".join("?" for _ in objective_ids)
        with self.database.read() as connection:
            rows = connection.execute(
                f"""SELECT decision.question_objective_id AS objective_id,
                           COUNT(DISTINCT attempt.family_id) AS families
                    FROM attempts attempt
                    JOIN decisions decision ON decision.id = attempt.decision_id
                    WHERE attempt.learner_id = ?
                      AND decision.question_objective_id IN ({placeholders})
                    GROUP BY decision.question_objective_id""",
                (learner_id, *sorted(objective_ids)),
            ).fetchall()
        for row in rows:
            objective_id = row["objective_id"]
            family_count = row["families"]
            if objective_id not in counts or type(family_count) is not int:
                raise ValidationError(
                    "Observed objective-family projection is inconsistent."
                )
            counts[objective_id] = family_count
        return counts

    @staticmethod
    def _persistent_gap_marker(
        *,
        rationale: str,
        policy_version: str,
        decision_objective_id: str | None,
    ) -> dict[str, object] | None:
        """Parse one durable episode spend marker, failing closed on drift."""

        if type(rationale) is not str:
            raise ValidationError(
                "Persistent-gap episode rationale must be a string."
            )
        terms: dict[str, str] = {}
        for raw_term in rationale.split(";"):
            term = raw_term.strip()
            if not term.startswith("persistent_gap_"):
                continue
            if "=" not in term:
                raise ValidationError(
                    "Malformed persistent-gap episode rationale marker."
                )
            key, value = term.split("=", 1)
            if (
                key in terms
                or key
                not in _PERSISTENT_GAP_BASE_MARKERS
                | _PERSISTENT_GAP_EPISODE_MARKERS
            ):
                raise ValidationError(
                    "Malformed persistent-gap episode rationale marker."
                )
            terms[key] = value
        if not terms:
            return None
        if not _PERSISTENT_GAP_BASE_MARKERS <= set(terms):
            raise ValidationError(
                "Persistent-gap episode rationale is missing required markers."
            )
        objective_id = terms["persistent_gap_revisit"]
        if (
            type(decision_objective_id) is not str
            or not decision_objective_id
            or objective_id != decision_objective_id
        ):
            raise ValidationError(
                "Persistent-gap episode marker crossed learning objectives."
            )

        try:
            observed_families = int(
                terms["persistent_gap_observed_families"]
            )
            mastery = float(terms["persistent_gap_mastery"])
            cold_prior = float(terms["persistent_gap_cold_prior"])
            due_at = datetime.fromisoformat(
                terms["persistent_gap_due_at"]
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError(
                "Persistent-gap episode rationale has invalid values."
            ) from exc
        if (
            str(observed_families)
            != terms["persistent_gap_observed_families"]
            or observed_families < PERSISTENT_GAP_MIN_OBSERVED_FAMILIES
            or not math.isfinite(mastery)
            or not math.isfinite(cold_prior)
            or not 0.0 <= mastery <= 1.0
            or not 0.0 <= cold_prior <= 1.0
            or cold_prior <= mastery
            or due_at.tzinfo is None
            or due_at.utcoffset() is None
        ):
            raise ValidationError(
                "Persistent-gap episode rationale has invalid values."
            )

        episode_terms = set(terms) & _PERSISTENT_GAP_EPISODE_MARKERS
        if episode_terms:
            if (
                policy_version
                not in PERSISTENT_GAP_EPISODE_POLICY_VERSIONS
            ):
                raise ValidationError(
                    "Unsupported persistent-gap episode policy version."
                )
            if episode_terms != _PERSISTENT_GAP_EPISODE_MARKERS:
                raise ValidationError(
                    "Persistent-gap episode rationale is missing its budget marker."
                )
            try:
                spend = int(terms["persistent_gap_episode_spend"])
                budget = int(terms["persistent_gap_episode_budget"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValidationError(
                    "Persistent-gap episode rationale has invalid budget values."
                ) from exc
            if (
                str(spend) != terms["persistent_gap_episode_spend"]
                or str(budget) != terms["persistent_gap_episode_budget"]
                or budget != PERSISTENT_GAP_EPISODE_BUDGET
                or not 1 <= spend <= budget
            ):
                raise ValidationError(
                    "Persistent-gap episode rationale has invalid budget values."
                )
        else:
            if policy_version == POLICY_VERSION:
                raise ValidationError(
                    "Current persistent-gap rationale lacks episode markers."
                )
            # v12 introduced the complete base marker before bounded episodes.
            # It can safely seed spend one when an active session crosses the
            # policy upgrade boundary.
            if policy_version != "recursive-evidence-graph-v12":
                raise ValidationError(
                    "Unsupported legacy persistent-gap rationale version."
                )
            spend = 1
            budget = PERSISTENT_GAP_EPISODE_BUDGET
        return {
            "objective_id": objective_id,
            "observed_response_families": observed_families,
            "mastery_probability": mastery,
            "cold_start_mastery_probability": cold_prior,
            "opened_due_at": due_at,
            "spend": spend,
            "budget": budget,
        }

    def _persistent_gap_episode_history(
        self,
        *,
        session_id: str,
        learner_id: str,
    ) -> dict[str, dict[str, object]]:
        """Reconstruct bounded episode spends from immutable response order."""

        expected_stream_id = f"learner:{learner_id}"
        with self.database.read() as connection:
            rows = connection.execute(
                """SELECT decision.question_objective_id,
                          decision.rationale, decision.policy_version,
                          attempt.family_id, event.stream_id,
                          event.stream_version
                   FROM attempts attempt
                   JOIN decisions decision
                     ON decision.id = attempt.decision_id
                   JOIN events event ON event.event_id = attempt.event_id
                   WHERE attempt.session_id = ?
                     AND attempt.learner_id = ?
                     AND event.event_type = 'ResponseSubmitted'
                   ORDER BY event.stream_version""",
                (session_id, learner_id),
            ).fetchall()

        histories: dict[str, dict[str, object]] = {}
        processed: list[tuple[int, str | None]] = []
        for row in rows:
            if row["stream_id"] != expected_stream_id:
                raise ValidationError(
                    "Persistent-gap episode history crossed learner streams."
                )
            marker = self._persistent_gap_marker(
                rationale=row["rationale"],
                policy_version=row["policy_version"],
                decision_objective_id=row["question_objective_id"],
            )
            if marker is not None:
                objective_id = marker["objective_id"]
                history = histories.setdefault(
                    objective_id,
                    {
                        "spends": 0,
                        "family_ids": set(),
                        "opened_due_at": marker["opened_due_at"],
                        "last_spend_stream_version": None,
                    },
                )
                if (
                    history["opened_due_at"].astimezone(timezone.utc)
                    != marker["opened_due_at"].astimezone(timezone.utc)
                ):
                    raise ValidationError(
                        "Persistent-gap episode changed its opening due timestamp."
                    )
                expected_spend = int(history["spends"]) + 1
                spend = int(marker["spend"])
                if spend != expected_spend:
                    raise ValidationError(
                        "Persistent-gap episode spend sequence is malformed."
                    )
                family_ids = history["family_ids"]
                if row["family_id"] in family_ids:
                    raise ValidationError(
                        "Persistent-gap episode reused a response family."
                    )
                if spend > 1:
                    previous_version = history[
                        "last_spend_stream_version"
                    ]
                    if not any(
                        stream_version > previous_version
                        and response_objective_id != objective_id
                        for stream_version, response_objective_id in processed
                    ):
                        raise ValidationError(
                            "Persistent-gap episode spends were not interleaved."
                        )
                family_ids.add(row["family_id"])
                history["spends"] = spend
                history["last_spend_stream_version"] = row[
                    "stream_version"
                ]
            processed.append(
                (row["stream_version"], row["question_objective_id"])
            )

        for objective_id, history in histories.items():
            last_spend = history["last_spend_stream_version"]
            history["interleaved_since_last_spend"] = any(
                stream_version > last_spend
                and response_objective_id != objective_id
                for stream_version, response_objective_id in processed
            )
            history["family_ids"] = frozenset(history["family_ids"])
        return histories

    @staticmethod
    def _next_persistent_gap_episode_spend(
        *,
        prior_spends: int,
        gap_open: bool,
        due: bool,
        interleaved: bool,
        distinct_capacity: bool,
    ) -> int | None:
        """Return the next bounded spend number, or close/block the episode."""

        if (
            type(prior_spends) is not int
            or not 0 <= prior_spends <= PERSISTENT_GAP_EPISODE_BUDGET
            or any(
                type(value) is not bool
                for value in (
                    gap_open,
                    due,
                    interleaved,
                    distinct_capacity,
                )
            )
        ):
            raise ValidationError(
                "Persistent-gap episode state is malformed."
            )
        if (
            not gap_open
            or not distinct_capacity
            or prior_spends >= PERSISTENT_GAP_EPISODE_BUDGET
        ):
            return None
        if prior_spends == 0:
            return 1 if due else None
        return 2 if interleaved else None

    @staticmethod
    def _coverage_target(question: Question) -> tuple[str, str]:
        if question.objective_id is not None:
            if (
                type(question.objective_id) is not str
                or not question.objective_id
            ):
                raise ValidationError(
                    "A hybrid-coverage objective target is malformed."
                )
            return ("objective", question.objective_id)
        if (
            type(question.primary_concept_id) is not str
            or not question.primary_concept_id
        ):
            raise ValidationError(
                "A hybrid-coverage concept target is malformed."
            )
        return ("concept", question.primary_concept_id)

    def _successful_retrieval_family_counts(
        self,
        learner_id: str,
        release_id: str,
        targets: set[tuple[str, str]],
    ) -> dict[tuple[str, str], int]:
        """Read positive families accepted by the pinned release.

        Removed, quarantined, or emergency-revoked item families cannot
        contribute current internal retrieval coverage.
        """

        if type(learner_id) is not str or not learner_id:
            raise ValidationError(
                "Hybrid coverage requires a valid learner identifier."
            )
        if type(release_id) is not str or not release_id:
            raise ValidationError(
                "Hybrid coverage requires a valid pinned release identifier."
            )
        counts = {target: 0 for target in targets}
        if not targets:
            return counts
        families = {target: set() for target in targets}
        with self.database.read() as connection:
            release = connection.execute(
                """SELECT sealed_at FROM corpus_releases WHERE id = ?""",
                (release_id,),
            ).fetchone()
            if release is None or release["sealed_at"] is None:
                raise ValidationError(
                    "Hybrid coverage requires an existing sealed pinned release."
                )
            rows = connection.execute(
                """SELECT 'objective' AS target_kind,
                          evidence.objective_id AS target_id,
                          evidence.family_id
                   FROM learner_objective_families evidence
                   WHERE evidence.learner_id = ?
                     AND EXISTS (
                         SELECT 1
                         FROM release_question_objectives direct
                         JOIN questions question
                           ON question.id = direct.question_id
                         JOIN release_questions released
                           ON released.release_id = direct.release_id
                          AND released.question_id = direct.question_id
                         WHERE direct.release_id = ?
                           AND direct.objective_id = evidence.objective_id
                           AND question.family_id = evidence.family_id
                           AND released.status IN ('approved', 'calibrated')
                           AND NOT EXISTS (
                               SELECT 1
                               FROM question_revocations revoked
                               WHERE revoked.question_id = question.id
                           )
                     )
                   UNION ALL
                   SELECT 'concept' AS target_kind,
                          evidence.concept_id AS target_id,
                          evidence.family_id
                   FROM learner_skill_families evidence
                   WHERE evidence.learner_id = ?
                     AND EXISTS (
                         SELECT 1
                         FROM release_questions released
                         JOIN questions question
                           ON question.id = released.question_id
                         JOIN question_concepts mapping
                           ON mapping.question_id = question.id
                          AND mapping.role = 'primary'
                         WHERE released.release_id = ?
                           AND released.status IN ('approved', 'calibrated')
                           AND mapping.concept_id = evidence.concept_id
                           AND question.family_id = evidence.family_id
                           AND NOT EXISTS (
                               SELECT 1
                               FROM question_revocations revoked
                               WHERE revoked.question_id = question.id
                           )
                     )
                   ORDER BY target_kind, target_id, family_id""",
                (learner_id, release_id, learner_id, release_id),
            ).fetchall()
        for row in rows:
            target = (row["target_kind"], row["target_id"])
            family_id = row["family_id"]
            if (
                target[0] not in {"objective", "concept"}
                or type(target[1]) is not str
                or not target[1]
                or type(family_id) is not str
                or not family_id
            ):
                raise ValidationError(
                    "Hybrid coverage found a malformed retrieval-family ledger."
                )
            if target not in families:
                continue
            if family_id in families[target]:
                raise ValidationError(
                    "Hybrid coverage found a duplicate retrieval family."
                )
            families[target].add(family_id)
        for target, target_families in families.items():
            counts[target] = len(target_families)
        return counts

    def _hybrid_coverage_by_question(
        self,
        questions: Iterable[Question],
        *,
        learner_id: str,
        release_id: str,
        objective_states: Mapping[str, ObjectiveState],
        concept_states: Mapping[str, object],
    ) -> dict[str, _HybridCoverage]:
        """Resolve exact pre-selection burden, information, and retrieval state."""

        candidates = tuple(questions)
        question_ids = [question.id for question in candidates]
        if (
            any(
                type(question_id) is not str or not question_id
                for question_id in question_ids
            )
            or len(question_ids) != len(set(question_ids))
        ):
            raise ValidationError(
                "Hybrid coverage requires unique non-empty question IDs."
            )
        targets = {
            self._coverage_target(question) for question in candidates
        }
        retrieval_counts = self._successful_retrieval_family_counts(
            learner_id, release_id, targets
        )
        by_target: dict[tuple[str, str], _HybridCoverage] = {}
        for target_kind, target_id in sorted(targets):
            if target_kind == "objective":
                state = objective_states.get(target_id)
                state_identity = (
                    state.objective_id if state is not None else target_id
                )
            else:
                state = concept_states.get(target_id)
                state_identity = (
                    state.concept_id if state is not None else target_id
                )
            if state_identity != target_id:
                raise ValidationError(
                    "Hybrid coverage projection crossed adaptive targets."
                )
            raw_exposures = state.exposures if state is not None else 0
            diagnostic_information = (
                state.evidence_mass if state is not None else 0.0
            )
            by_target[(target_kind, target_id)] = _HybridCoverage(
                raw_exposures=raw_exposures,
                diagnostic_information=diagnostic_information,
                successful_retrieval_families=retrieval_counts[
                    (target_kind, target_id)
                ],
            )
        return {
            question.id: by_target[self._coverage_target(question)]
            for question in candidates
        }

    @staticmethod
    def _fair_coverage_candidates(
        questions: Iterable[Question],
        *,
        coverage_by_question: Mapping[str, _HybridCoverage],
        persistent_gap_objective_ids: set[str],
    ) -> tuple[list[Question], int, set[str]]:
        """Apply hybrid breadth without confusing burden, evidence, and skill."""

        candidates = list(questions)
        if not candidates:
            raise ValidationError(
                "Fair-coverage selection requires at least one candidate."
            )
        if any(
            question.id not in coverage_by_question
            or not isinstance(
                coverage_by_question[question.id], _HybridCoverage
            )
            for question in candidates
        ):
            raise ValidationError(
                "Fair-coverage hybrid state is incomplete or invalid."
            )
        minimum = min(
            coverage_by_question[question.id].raw_exposures
            for question in candidates
        )
        if minimum == 0:
            # Complete an initial breadth sweep before a previously observed
            # persistent gap may bypass an entirely unseen target.
            ordinary = [
                question
                for question in candidates
                if coverage_by_question[question.id].raw_exposures == 0
            ]
            persistent_candidates: list[Question] = []
        else:
            burden_limit = minimum + HYBRID_COVERAGE_RAW_BURDEN_SLACK
            burden_safe = [
                question
                for question in candidates
                if coverage_by_question[question.id].raw_exposures
                <= burden_limit
            ]
            minimum_retrieval = min(
                coverage_by_question[
                    question.id
                ].successful_retrieval_families
                for question in burden_safe
            )
            retrieval_frontier = [
                question
                for question in burden_safe
                if coverage_by_question[
                    question.id
                ].successful_retrieval_families
                == minimum_retrieval
            ]
            minimum_information = min(
                coverage_by_question[
                    question.id
                ].diagnostic_information
                for question in retrieval_frontier
            )
            ordinary = [
                question
                for question in retrieval_frontier
                if coverage_by_question[
                    question.id
                ].diagnostic_information
                == minimum_information
            ]
            persistent_candidates = [
                question
                for question in candidates
                if question.objective_id in persistent_gap_objective_ids
            ]
        selected_ids = {
            question.id for question in (*ordinary, *persistent_candidates)
        }
        selected = [
            question for question in candidates if question.id in selected_ids
        ]
        bypassed = {
            question.objective_id
            for question in persistent_candidates
            if coverage_by_question[question.id].raw_exposures > minimum
        }
        return selected, minimum, {
            objective_id for objective_id in bypassed if objective_id is not None
        }

    def choose(self, session_id: str, *, now: datetime | None = None) -> Presentation:
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValidationError("now must be timezone-aware.")
        now = now.astimezone(timezone.utc)
        exploration_fallback: tuple[str, tuple[str, ...]] | None = None
        retries = 0
        while retries < 4:
            try:
                return self._choose_once(
                    session_id,
                    now=now,
                    allow_exploration=exploration_fallback is None,
                    exploration_fallback=exploration_fallback,
                )
            except _ExplorationUnserviceable as exc:
                if exploration_fallback is not None:
                    raise ValidationError(
                        "Requested-scope fallback attempted exploration twice."
                    ) from exc
                exploration_fallback = (exc.reason, exc.topic_ids)
            except _RetrySelection:
                retries += 1
                continue
        raise ExhaustedError("Selection state changed repeatedly; retry the request.")

    def _choose_once(
        self,
        session_id: str,
        *,
        now: datetime,
        allow_exploration: bool = True,
        exploration_fallback: tuple[str, tuple[str, ...]] | None = None,
    ) -> Presentation:
        session = self.database.get_session(session_id)
        if session["status"] != "active":
            raise ExhaustedError(f"Session {session_id} is {session['status']}.")
        self.database.validate_session_focus(session)
        # A choice is valid only for the learner projection against which it was
        # selected.  Another active session may advance that projection before
        # this session answers, so reconcile the pending choice under the write
        # lock before either returning it or doing expensive candidate scoring.
        current_pending = False
        with self.database.transaction() as connection:
            current_session = connection.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            learner_row = connection.execute(
                "SELECT revision FROM learners WHERE id = ?", (session["learner_id"],)
            ).fetchone()
            if (
                not current_session
                or current_session["status"] != "active"
                or current_session["revision"] != session["revision"]
                or not learner_row
            ):
                raise _RetrySelection()
            self.database.require_learner_evidence_safe(
                session["learner_id"],
                connection,
            )
            open_performance_attempt = connection.execute(
                """SELECT attempt.id
                   FROM performance_attempts attempt
                   WHERE attempt.session_id = ?
                     AND NOT EXISTS (
                         SELECT 1 FROM performance_actions terminal
                         WHERE terminal.attempt_id = attempt.id
                           AND terminal.action_type IN ('submitted', 'abandoned')
                     )
                   ORDER BY attempt.started_at, attempt.id LIMIT 1""",
                (session_id,),
            ).fetchone()
            if open_performance_attempt is not None:
                raise ConflictError(
                    "Session has an active performance task "
                    f"{open_performance_attempt['id']}; submit or abandon it "
                    "before selecting another question."
                )
            learner_revision = learner_row["revision"]
            pending_row = connection.execute(
                """SELECT decision.*, revocation.revoked_at AS emergency_revoked_at
                   FROM decisions decision
                   LEFT JOIN question_revocations revocation
                     ON revocation.question_id = decision.question_id
                   WHERE decision.session_id = ?
                     AND decision.consumed_at IS NULL
                     AND decision.invalidated_at IS NULL
                   ORDER BY decision.created_at DESC LIMIT 1""",
                (session_id,),
            ).fetchone()
            selection_boundary = (
                _selection_version_boundary(
                    connection,
                    decision_id=pending_row["id"],
                    session_id=session_id,
                )
                if pending_row
                else None
            )
            selection_model_version = (
                selection_boundary[0] if selection_boundary else None
            )
            selection_policy_version = (
                selection_boundary[1] if selection_boundary else None
            )
            pending_activation_safe = True
            if pending_row is not None:
                pending_question = self.database.get_question(
                    pending_row["question_id"],
                    connection,
                    release_id=pending_row["corpus_release_id"],
                )
                pending_activation_safe = question_runtime_activation_safe(
                    pending_question,
                    status=pending_row["question_status"],
                )
            if (
                pending_row
                and selection_policy_version
                != pending_row["policy_version"]
            ):
                raise ValidationError(
                    f"Pending decision {pending_row['id']} policy does not "
                    "match its QuestionSelected boundary."
                )
            if (
                pending_row
                and pending_row["emergency_revoked_at"] is None
                and pending_activation_safe
                and pending_row["learner_revision"] == learner_revision
                and selection_model_version == self.learner_model.model_version
                and selection_policy_version == POLICY_VERSION
            ):
                current_pending = True
            elif pending_row:
                reason = (
                    "question_emergency_revoked"
                    if pending_row["emergency_revoked_at"] is not None
                    else (
                        "question_activation_provenance_invalid"
                        if not pending_activation_safe
                        else (
                            "learner_model_changed"
                            if selection_model_version
                            != self.learner_model.model_version
                            else (
                                "policy_changed"
                                if selection_policy_version != POLICY_VERSION
                                else "learner_projection_advanced"
                            )
                        )
                    )
                )
                invalidated = connection.execute(
                    """UPDATE decisions
                       SET invalidated_at = ?, invalidation_reason = ?
                       WHERE id = ? AND consumed_at IS NULL
                         AND invalidated_at IS NULL""",
                    (now.isoformat(), reason, pending_row["id"]),
                )
                if invalidated.rowcount != 1:
                    raise _RetrySelection()
                self.database.append_event(
                    connection,
                    stream_id=f"learner:{session['learner_id']}",
                    event_type="DecisionInvalidated",
                    payload={
                        "decision_id": pending_row["id"],
                        "reason": reason,
                        "selection_learner_revision": pending_row["learner_revision"],
                        "current_learner_revision": learner_revision,
                    },
                    metadata={
                        "policy_version": POLICY_VERSION,
                        "learner_model_version": self.learner_model.model_version,
                        "corpus_release_id": pending_row["corpus_release_id"],
                    },
                    learner_id=session["learner_id"],
                    session_id=session_id,
                    causation_id=pending_row["id"],
                    occurred_at=now,
                )
        if current_pending:
            pending = self.database.pending_presentation(session_id)
            if pending:
                if (
                    pending.question.objective_id is not None
                    and self.learner_model.model_version
                    not in OBJECTIVE_MODEL_VERSIONS
                ):
                    raise ValidationError(
                        f"Learner model {self.learner_model.model_version} cannot "
                        "serve an objective-aware pending question; use the "
                        "current learner model."
                    )
                return pending
            raise _RetrySelection()
        release_id = session["corpus_release_id"]
        phase = SessionPhase(session["phase"])
        graph = self.database.get_graph(release_id)
        topic_id = session.get("topic_id")
        if topic_id:
            base_scope = self.database.topic_scope(topic_id, release_id)
            owned_targets = self.database.topic_owned_concepts(
                topic_id, release_id, include_descendants=True
            )
        else:
            base_scope = graph.learning_scope(session["root_concept_id"])
            owned_targets = {session["root_concept_id"]}
        scope = set(base_scope)
        # A failed deliberate exploration can focus remediation outside the
        # requested topic. The focused tunnel must remain serviceable even
        # though ordinary main questions return to the requested curriculum.
        if session["focus_concept_id"]:
            scope.update(graph.learning_scope(session["focus_concept_id"]))

        recent_performance = self.database.session_recent_performance(
            session_id, limit=3
        )
        exploration_topic_ids: tuple[str, ...] = ()
        exploring = False
        if allow_exploration and self._should_explore(
            session, phase, recent_performance
        ):
            catalog = self.database.get_catalog(release_id)
            topic = next(
                (item for item in catalog["topics"] if item["id"] == topic_id),
                None,
            )
            if topic is not None:
                exploration_topic_ids = tuple(topic["related_topic_ids"])
                exploration_scope: set[str] = set()
                for related_topic_id in exploration_topic_ids:
                    exploration_scope.update(
                        self.database.topic_scope(related_topic_id, release_id)
                    )
                exploration_scope -= base_scope
                if exploration_scope:
                    scope = exploration_scope
                    exploring = True
        concepts = graph.concepts
        persisted_concept_states = self.database.get_skill_states(
            session["learner_id"]
        )
        objective_states = self.database.get_objective_states(
            session["learner_id"]
        )
        release_objectives = self.database.get_learning_objectives(release_id)
        cold_objective_states = self.learner_model.initial_objective_states(
            session["learner_id"], release_objectives
        )
        # Objective projection is substantially more expensive than looking up
        # a Gaussian concept state: v6 applies retention to a full fixed-grid
        # density and derives its metrics.  Every projection in this selection
        # shares one learner snapshot and one clock, so materialize it once per
        # release objective and reuse the immutable result below.
        projected_objective_states = {}
        for objective in release_objectives:
            objective_state = objective_states.get(objective.id)
            if objective_state is None:
                objective_state = cold_objective_states[objective.id]
            projected_objective_states[objective.id] = (
                self.learner_model.project_objective_state(
                    objective_state, objective, now
                )
            )
        projected_objective_states = MappingProxyType(
            projected_objective_states
        )
        floor_projection = (
            self.learner_model.concept_projection_with_objective_floor(
                learner_id=session["learner_id"],
                concepts=concepts,
                stored_states=persisted_concept_states,
                objectives=release_objectives,
                stored_objective_states=objective_states,
                now=now,
                projected_objective_states=projected_objective_states,
            )
        )
        stored_states = floor_projection.states
        focus_objective_id = session.get("focus_objective_id")
        target_ids = [session["focus_concept_id"]] if session["focus_concept_id"] else list(scope)
        target_means = []
        if focus_objective_id:
            focus_objective = next(
                (
                    objective
                    for objective in release_objectives
                    if objective.id == focus_objective_id
                ),
                None,
            )
            if focus_objective is None:
                raise ValidationError(
                    f"Focused learning objective {focus_objective_id} is not in "
                    f"release {release_id}."
                )
            # Objective items may use any declared supporting concept as their
            # primary retrieval mapping. Keep the complete immutable objective
            # scope reachable during its repair/verification episode.
            scope.update(focus_objective.concept_ids)
            target_means.append(
                projected_objective_states[focus_objective_id].mean
            )
        for concept_id in target_ids:
            if not concept_id or concept_id not in concepts:
                continue
            concept = concepts[concept_id]
            state = stored_states.get(concept_id) or self.learner_model.initial_state(
                session["learner_id"], concept
            )
            if not focus_objective_id:
                target_means.append(
                    self.learner_model.project_state(state, concept, now).mean
                )
        target_difficulty = median(target_means) if target_means else 0.0
        questions = self.database.questions_for_scope(
            scope,
            learner_id=session["learner_id"],
            focus_concept_id=session["focus_concept_id"],
            focus_misconception_id=session["focus_misconception_id"],
            focus_objective_id=focus_objective_id,
            release_id=release_id,
            target_difficulty=target_difficulty,
            limit=600,
        )
        main_candidate_ids = {question.id for question in questions}
        remediation_path = session.get("remediation_path") or []
        parent_obligation = (
            remediation_path[-1] if remediation_path else None
        )
        parent_objective_id = (
            parent_obligation.get("objective_id")
            if parent_obligation is not None
            else None
        )
        if (
            questions
            and focus_objective_id is not None
            and parent_objective_id is not None
        ):
            # Focused retrieval is normally exact to the child objective. A
            # pending parent obligation also needs its release-wide verification
            # reserve in the safety pool, although those parent questions remain
            # ineligible until the child has been repaired and verified.
            by_id = {question.id: question for question in questions}
            parent_reserve = self.database.questions_for_scope(
                set(),
                learner_id=session["learner_id"],
                focus_misconception_id=None,
                focus_objective_id=parent_objective_id,
                release_id=release_id,
                target_difficulty=target_difficulty,
                limit=600,
            )
            for reserve in parent_reserve:
                by_id.setdefault(reserve.id, reserve)
            questions = list(by_id.values())
        if questions and focus_objective_id is None:
            # Main selection is scope-bounded, but its safety proof is not.
            # Pull release-wide exact reserves for every objective a candidate
            # or distractor can activate so cross-primary repair families
            # cannot be hidden by the broad 600-item cutoff.
            reserve_objective_ids = {
                objective_id
                for question in questions
                for objective_id in (
                    question.objective_id,
                    *(
                        option.diagnostic_objective_id
                        for option in question.options
                    ),
                )
                if objective_id is not None
            }
            by_id = {question.id: question for question in questions}
            for objective_id in sorted(reserve_objective_ids):
                reserve_questions = self.database.questions_for_scope(
                    set(),
                    learner_id=session["learner_id"],
                    focus_misconception_id=None,
                    focus_objective_id=objective_id,
                    release_id=release_id,
                    target_difficulty=target_difficulty,
                    limit=600,
                )
                for reserve in reserve_questions:
                    by_id.setdefault(reserve.id, reserve)
            questions = list(by_id.values())
        if not questions:
            if exploring:
                raise _ExplorationUnserviceable(
                    "no_approved_questions", exploration_topic_ids
                )
            target = topic_id or session["root_concept_id"]
            raise ExhaustedError(f"Corpus gap: no approved questions cover {target}.")
        if (
            self.learner_model.model_version not in OBJECTIVE_MODEL_VERSIONS
            and any(question.objective_id is not None for question in questions)
        ):
            raise ValidationError(
                f"Learner model {self.learner_model.model_version} cannot select "
                "objective-aware questions; use the current learner model."
            )

        beliefs = self.database.get_misconception_beliefs(session["learner_id"])
        exposure = self.database.get_exposure_summary(
            session["learner_id"],
            question_ids={question.id for question in questions},
            family_ids={question.family_id for question in questions},
        )
        reuse_eligible: list[Question] = []
        for question in questions:
            family_exposure = exposure["families"].get(question.family_id)
            if family_exposure is None:
                reuse_eligible.append(question)
                continue
            try:
                family_last_at = datetime.fromisoformat(family_exposure["last_at"])
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"Family {question.family_id} has an invalid exposure "
                    "timestamp."
                ) from exc
            if family_last_at.tzinfo is None or family_last_at.utcoffset() is None:
                raise ValidationError(
                    f"Family {question.family_id} has a timezone-naive exposure "
                    "timestamp."
                )
            if question.objective is not None:
                projected = projected_objective_states[question.objective.id]
            else:
                concept = concepts[question.primary_concept_id]
                state = stored_states.get(question.primary_concept_id)
                if state is None:
                    state = self.learner_model.initial_state(
                        session["learner_id"], concept
                    )
                projected = self.learner_model.project_state(
                    state, concept, now
                )
            if now - family_last_at < self.learner_model.required_family_spacing(
                projected
            ):
                continue
            if phase == SessionPhase.REVIEW and (
                projected.next_review_at is None
                or projected.next_review_at > now
            ):
                continue
            reuse_eligible.append(question)
        questions = reuse_eligible
        if not questions:
            if exploring:
                raise _ExplorationUnserviceable(
                    "no_due_independent_family", exploration_topic_ids
                )
            raise ExhaustedError(
                "No safely independent question family is due yet for this learner."
            )
        if (
            phase == SessionPhase.REVIEW
            and not exploring
            and focus_objective_id is None
            and not any(
                question.id in main_candidate_ids
                and (
                    question.objective.primary_concept_id
                    if question.objective is not None
                    else question.primary_concept_id
                )
                in owned_targets
                for question in questions
            )
        ):
            # Release-wide objective reserves and questions attached through a
            # supporting concept can remain due even when nothing owned by the
            # requested curriculum is due.  They are not evidence of a corpus
            # deficit and must not enqueue an authoring job for the requested
            # topic.
            raise ExhaustedError(
                "No safely independent question family is due yet for the "
                "requested review target."
            )
        recent = list(session["recent_families"])
        focus_concept = session["focus_concept_id"]
        focus_misconception = session["focus_misconception_id"]
        focus_objective = session.get("focus_objective_id")

        session_exposure = self.database.session_exposure_summary(session_id)
        independent_pool = [
            question
            for question in questions
            if question.id not in session_exposure["questions"]
            and question.family_id not in session_exposure["families"]
            and question.family_id not in recent[-4:]
        ]
        if not independent_pool:
            if exploring:
                raise _ExplorationUnserviceable(
                    "no_unseen_independent_family", exploration_topic_ids
                )
            raise ExhaustedError(
                "Corpus gap: no unseen independent item family remains in this session."
            )
        candidate_pool = independent_pool
        parent_verification_families: set[str] = set()
        if parent_objective_id is not None:
            parent_misconception_id = parent_obligation.get(
                "misconception_id"
            )
            parent_verification_families = {
                question.family_id
                for question in independent_pool
                if question.objective_id == parent_objective_id
                and question.kind in VERIFICATION_KINDS
                and (
                    parent_misconception_id is None
                    or any(
                        option.misconception_id
                        == parent_misconception_id
                        and (
                            option.diagnostic_objective_id
                            or question.objective_id
                        )
                        == parent_objective_id
                        for option in question.options
                    )
                )
            }

        pedagogical_role = "exploration_probe" if exploring else "main"
        focus_valid = False
        active_misconception_revisit: str | None = None
        eligible = candidate_pool
        if phase == SessionPhase.REMEDIATE and (focus_concept or focus_misconception):
            verification_families = {
                question.family_id
                for question in independent_pool
                if (
                    (focus_objective and question.objective_id == focus_objective)
                    or (
                        not focus_objective
                        and focus_concept
                        and question.primary_concept_id == focus_concept
                    )
                )
                and question.kind in VERIFICATION_KINDS
                and (
                    not (focus_objective and focus_misconception)
                    or any(
                        option.misconception_id == focus_misconception
                        and (
                            option.diagnostic_objective_id
                            or question.objective_id
                        )
                        == focus_objective
                        for option in question.options
                    )
                )
            }
            if focus_misconception:
                probes = [
                    question
                    for question in independent_pool
                    if focus_misconception in question.misconception_ids
                    and (
                        not focus_objective
                        or (
                            question.objective_id == focus_objective
                            and any(
                                option.misconception_id == focus_misconception
                                and (
                                    option.diagnostic_objective_id
                                    or question.objective_id
                                )
                                == focus_objective
                                for option in question.options
                            )
                        )
                    )
                ]
            elif focus_objective:
                probes = [
                    question
                    for question in independent_pool
                    if question.objective_id == focus_objective
                ]
            else:
                probes = [
                    question
                    for question in independent_pool
                    if focus_concept
                    and question.primary_concept_id == focus_concept
                ]
            # Do not consume the last independent transfer family as the repair
            # probe; verification must remain possible before serving anything.
            # When this child has a parent obligation, reserve a third distinct
            # family for the parent's eventual transfer check as well.
            if parent_objective_id is not None:
                focused = [
                    question
                    for question in probes
                    if any(
                        parent_verification_families
                        - {
                            question.family_id,
                            child_verification_family,
                        }
                        for child_verification_family in (
                            verification_families
                            - {question.family_id}
                        )
                    )
                ]
            else:
                focused = [
                    question
                    for question in probes
                    if verification_families - {question.family_id}
                ]
            if not focused:
                reserve_clause = (
                    " while preserving an independent parent verification reserve"
                    if parent_objective_id is not None
                    else ""
                )
                raise ExhaustedError(
                    "Corpus gap: no remediation-plus-verification pair exists for "
                    f"{focus_misconception or focus_concept}{reserve_clause}. "
                    "Run `tsq coverage --enqueue`."
                )
            eligible = focused
            pedagogical_role = "remediation_probe"
            focus_valid = True
        elif phase == SessionPhase.VERIFY and (focus_concept or focus_objective):
            focused = [
                question
                for question in independent_pool
                if (
                    (focus_objective and question.objective_id == focus_objective)
                    or (
                        not focus_objective
                        and question.primary_concept_id == focus_concept
                    )
                )
                and question.kind in VERIFICATION_KINDS
                and (
                    not (focus_objective and focus_misconception)
                    or any(
                        option.misconception_id == focus_misconception
                        and (
                            option.diagnostic_objective_id
                            or question.objective_id
                        )
                        == focus_objective
                        for option in question.options
                    )
                )
            ]
            if parent_objective_id is not None:
                focused = [
                    question
                    for question in focused
                    if parent_verification_families
                    - {question.family_id}
                ]
            if not focused:
                reserve_clause = (
                    " while preserving the parent verification reserve"
                    if parent_objective_id is not None
                    else ""
                )
                raise ExhaustedError(
                    "Corpus gap: no independent verification item exists for "
                    f"{focus_objective or focus_concept}{reserve_clause}. "
                    "Run `tsq coverage --enqueue`."
                )
            eligible = focused
            pedagogical_role = "verification"
            focus_valid = True
        elif phase in {
            SessionPhase.LEARN,
            SessionPhase.DIAGNOSE,
            SessionPhase.REVIEW,
        }:
            families_by_concept: dict[str, set[str]] = {}
            families_by_misconception: dict[str, set[str]] = {}
            families_by_objective: dict[str, set[str]] = {}
            families_by_objective_misconception: dict[
                tuple[str, str], set[str]
            ] = {}
            verification_by_concept: dict[str, set[str]] = {}
            verification_by_objective: dict[str, set[str]] = {}
            verification_by_objective_misconception: dict[
                tuple[str, str], set[str]
            ] = {}
            for question in independent_pool:
                families_by_concept.setdefault(
                    question.primary_concept_id, set()
                ).add(question.family_id)
                if question.kind in VERIFICATION_KINDS:
                    verification_by_concept.setdefault(
                        question.primary_concept_id, set()
                    ).add(question.family_id)
                if question.objective_id:
                    families_by_objective.setdefault(
                        question.objective_id, set()
                    ).add(question.family_id)
                    if question.kind in VERIFICATION_KINDS:
                        verification_by_objective.setdefault(
                            question.objective_id, set()
                        ).add(question.family_id)
                for option in question.options:
                    misconception_id = option.misconception_id
                    if misconception_id is None:
                        continue
                    families_by_misconception.setdefault(
                        misconception_id, set()
                    ).add(question.family_id)
                    if (
                        question.objective_id
                        and (
                            option.diagnostic_objective_id
                            or question.objective_id
                        )
                        == question.objective_id
                    ):
                        families_by_objective_misconception.setdefault(
                            (question.objective_id, misconception_id), set()
                        ).add(question.family_id)
                        if question.kind in VERIFICATION_KINDS:
                            verification_by_objective_misconception.setdefault(
                                (question.objective_id, misconception_id), set()
                            ).add(question.family_id)
            misconception_owners = {
                item.id: item.concept_id
                for item in self.database.get_misconceptions(
                    {
                        misconception_id
                        for question in independent_pool
                        for misconception_id in question.misconception_ids
                    },
                    release_id=release_id
                )
            }

            def serviceable(question: Question) -> bool:
                trigger_family = question.family_id
                primary = question.primary_concept_id
                objective_id = question.objective_id
                # A skipped/uncategorized error still needs one repair and one
                # independent transfer family after the trigger.
                primary_repairs = (
                    families_by_objective.get(objective_id, set())
                    if objective_id
                    else families_by_concept.get(primary, set())
                ) - {trigger_family}
                primary_verifications = (
                    verification_by_objective.get(objective_id, set())
                    if objective_id
                    else verification_by_concept.get(primary, set())
                ) - {trigger_family}
                if not any(
                    primary_verifications - {repair_family}
                    for repair_family in primary_repairs
                ):
                    return False
                for option in question.options:
                    misconception_id = option.misconception_id
                    if misconception_id is None:
                        continue
                    diagnostic_objective_id = (
                        option.diagnostic_objective_id or objective_id
                    )
                    owner = misconception_owners.get(misconception_id, primary)
                    repair_families = (
                        families_by_objective_misconception.get(
                            (diagnostic_objective_id, misconception_id), set()
                        )
                        if diagnostic_objective_id
                        else families_by_misconception.get(
                            misconception_id, set()
                        )
                    ) - {trigger_family}
                    verification_families = (
                        verification_by_objective_misconception.get(
                            (diagnostic_objective_id, misconception_id), set()
                        )
                        if diagnostic_objective_id
                        else verification_by_concept.get(owner, set())
                    ) - {trigger_family}
                    if (
                        objective_id is not None
                        and diagnostic_objective_id is not None
                        and diagnostic_objective_id != objective_id
                    ):
                        # A cross-objective diagnosis creates a four-family
                        # obligation: trigger A, repair B/m, verify B/m, then
                        # independently recheck transfer at A. Reserve the
                        # entire sequence simultaneously; pairwise checks can
                        # otherwise reuse the only A verification family as a
                        # B probe and strand the parent after child repair.
                        parent_verifications = (
                            verification_by_objective.get(objective_id, set())
                            - {trigger_family}
                        )
                        cross_objective_sequence_exists = any(
                            parent_verifications
                            - {
                                repair_family,
                                diagnostic_verification_family,
                            }
                            for repair_family in repair_families
                            for diagnostic_verification_family in (
                                verification_families - {repair_family}
                            )
                        )
                        if not cross_objective_sequence_exists:
                            return False
                    elif not any(
                        verification_families - {repair_family}
                        for repair_family in repair_families
                    ):
                        return False
                return True

            safe = [
                question
                for question in independent_pool
                if question.id in main_candidate_ids
                and serviceable(question)
            ]
            if not safe:
                if exploring:
                    raise _ExplorationUnserviceable(
                        "no_safely_serviceable_question",
                        exploration_topic_ids,
                    )
                raise ExhaustedError(
                    "Corpus gap: no safely serviceable main question remains while preserving "
                    "independent repair and verification families."
                )
            eligible = safe
            if not exploring:
                direct_requested_items = [
                    question
                    for question in eligible
                    if (
                        question.objective.primary_concept_id
                        if question.objective is not None
                        else question.primary_concept_id
                    )
                    in owned_targets
                ]
                if not direct_requested_items:
                    requested_target = (
                        f"curriculum topic {topic_id}"
                        if topic_id
                        else f"requested concept {session['root_concept_id']}"
                    )
                    raise ExhaustedError(
                        "Corpus gap: no safely serviceable question remains for "
                        f"{requested_target}."
                    )
                # Prerequisites remain in the modeled scope and can become a
                # focused remediation target after evidence warrants descent.
                # They are not silently substituted for the topic or concept
                # the learner explicitly selected.
                eligible = direct_requested_items

            # Treat a named misconception as a falsifiable hypothesis, not a
            # mastery verdict.  At most once, at the opening of a new session,
            # prefer a safely serviceable probe for the strongest active
            # in-scope hypothesis.  After that probe, ordinary breadth resumes
            # unless the response itself opens explicit remediation.
            if session["step"] == 0 and phase in {
                SessionPhase.LEARN,
                SessionPhase.DIAGNOSE,
            }:
                active_hypotheses = sorted(
                    (
                        (belief.probability, misconception_id)
                        for misconception_id, belief in beliefs.items()
                        if belief.probability
                        >= ACTIVE_MISCONCEPTION_REVISIT_THRESHOLD
                    ),
                    key=lambda item: (-item[0], item[1]),
                )
                for _, misconception_id in active_hypotheses:
                    probes = [
                        question
                        for question in eligible
                        if misconception_id in question.misconception_ids
                    ]
                    if probes:
                        eligible = probes
                        active_misconception_revisit = misconception_id
                        break

        hybrid_coverage_by_question = self._hybrid_coverage_by_question(
            eligible,
            learner_id=session["learner_id"],
            release_id=release_id,
            objective_states=objective_states,
            concept_states=persisted_concept_states,
        )
        fair_coverage_exposure: int | None = None
        persistent_gap_revisit_details: dict[str, dict[str, object]] = {}
        if (
            phase in {SessionPhase.LEARN, SessionPhase.DIAGNOSE}
            and not exploring
            and not focus_concept
            and not focus_objective
            and active_misconception_revisit is None
        ):
            episode_history = self._persistent_gap_episode_history(
                session_id=session_id,
                learner_id=session["learner_id"],
            )

            eligible_objective_ids = {
                question.objective_id
                for question in eligible
                if question.objective_id is not None
            }
            observed_objective_families = (
                self._observed_objective_response_families(
                    session["learner_id"], eligible_objective_ids
                )
            )
            objective_by_id = {
                objective.id: objective for objective in release_objectives
            }
            qualified_persistent_gaps: dict[str, dict[str, object]] = {}
            blocked_episode_objective_ids: set[str] = set()
            for objective_id in sorted(eligible_objective_ids):
                objective = objective_by_id.get(objective_id)
                projected = projected_objective_states.get(objective_id)
                observed_families = observed_objective_families[objective_id]
                if objective is None or projected is None:
                    raise ValidationError(
                        "A fair-coverage objective is missing from its pinned "
                        f"release projection: {objective_id}."
                    )
                if (
                    observed_families
                    >= PERSISTENT_GAP_MIN_OBSERVED_FAMILIES
                    and objective_id not in objective_states
                ):
                    raise ValidationError(
                        "Observed objective-family evidence lacks a durable "
                        f"objective projection: {objective_id}."
                    )
                cold_start = cold_objective_states[objective.id]
                history = episode_history.get(objective_id)
                prior_spends = (
                    int(history["spends"]) if history is not None else 0
                )
                previous_families = (
                    history["family_ids"]
                    if history is not None
                    else frozenset()
                )
                distinct_capacity = bool(
                    {
                        question.family_id
                        for question in eligible
                        if question.objective_id == objective_id
                    }
                    - previous_families
                )
                gap_open = self._has_persistent_gap(
                    projected,
                    cold_start,
                    observed_response_families=observed_families,
                )
                due = self._is_due_persistent_gap(
                    projected,
                    cold_start,
                    observed_response_families=observed_families,
                    now=now,
                )
                next_spend = self._next_persistent_gap_episode_spend(
                    prior_spends=prior_spends,
                    gap_open=gap_open,
                    due=due,
                    interleaved=bool(
                        history is not None
                        and history["interleaved_since_last_spend"]
                    ),
                    distinct_capacity=distinct_capacity,
                )
                if history is not None and next_spend is None:
                    # Once an episode opens, its target cannot leak back through
                    # the ordinary minimum-exposure frontier. It must satisfy
                    # the gap, capacity, spacing, interleaving, and budget
                    # contract or yield to another objective.
                    blocked_episode_objective_ids.add(objective_id)
                if next_spend is not None:
                    opened_due_at = (
                        history["opened_due_at"]
                        if history is not None
                        else projected.next_review_at
                    )
                    if not isinstance(opened_due_at, datetime):
                        raise ValidationError(
                            "A persistent-gap episode lacks its opening due time."
                        )
                    qualified_persistent_gaps[objective_id] = {
                        "observed_response_families": observed_families,
                        "mastery_probability": projected.mastery_probability,
                        "cold_start_mastery_probability": (
                            cold_start.mastery_probability
                        ),
                        "next_review_at": opened_due_at,
                        "episode_spend": next_spend,
                        "episode_budget": PERSISTENT_GAP_EPISODE_BUDGET,
                    }

            eligible = [
                question
                for question in eligible
                if question.objective_id
                not in blocked_episode_objective_ids
            ]
            if not eligible:
                raise ExhaustedError(
                    "No breadth or bounded persistent-gap candidate remains "
                    "after enforcing the session episode contract."
                )

            # Main-path adaptivity still handles a weakness immediately through
            # focused remediation. Between such episodes, maintain a hard
            # breadth frontier. A due objective supported by at least two
            # independent response families opens one explicitly bounded
            # revisit episode. Its second token survives the first response's
            # review-clock update, but only after a non-target response and only
            # while the exact gap and a distinct safe family remain.
            (
                eligible,
                fair_coverage_exposure,
                _,
            ) = self._fair_coverage_candidates(
                eligible,
                coverage_by_question=hybrid_coverage_by_question,
                persistent_gap_objective_ids=set(qualified_persistent_gaps),
            )
            # Mark a qualified spend even when that objective happened to sit
            # on the ordinary breadth frontier. Otherwise the first response
            # would be indistinguishable from ordinary coverage and could erase
            # the second bounded token by moving next_review_at.
            persistent_gap_revisit_details = qualified_persistent_gaps

        # Across sessions, prefer genuinely independent evidence before reusing
        # a family for the same primary concept.  Review is intentionally
        # exempt: spaced retrieval sometimes needs the previously learned
        # family, and its due-date signal should remain authoritative.
        if phase != SessionPhase.REVIEW:
            least_exposed = self._least_exposed_families_by_primary(
                eligible, exposure
            )
            least_exposed_ids = {
                question.id for question in least_exposed
            }
            eligible = [
                question
                for question in eligible
                if question.id in least_exposed_ids
                or question.objective_id
                in persistent_gap_revisit_details
            ]

        prerequisite_distances: dict[str, int] = {}
        for target_id in owned_targets:
            for concept_id, distance in graph.learning_distances_to(
                target_id
            ).items():
                prior = prerequisite_distances.get(concept_id)
                if prior is None or distance < prior:
                    prerequisite_distances[concept_id] = distance

        topic_by_concept: dict[str, str] = {}
        if topic_id:
            for catalog_topic in self.database.get_catalog(release_id)["topics"]:
                for concept in catalog_topic["concepts"]:
                    topic_by_concept[concept["id"]] = catalog_topic["id"]
        last_primary_concept = (
            recent_performance[0]["primary_concept_id"]
            if recent_performance
            else None
        )
        connected_pairs = {
            frozenset((edge.source_id, edge.target_id)) for edge in graph.edges
        }
        readiness = self.boundary_planner.readiness_map(
            learner_id=session["learner_id"],
            graph=graph,
            stored_states=stored_states,
            now=now,
            concept_ids={question.primary_concept_id for question in eligible},
            intrinsic_overrides=floor_projection.exact_floors,
        )
        potential_family_powers: dict[str, float] | None = None
        if (
            self.learner_model.model_version
            in SPACING_AWARE_FAMILY_MODEL_VERSIONS
        ):
            with self.database.read() as connection:
                potential = (
                    self.learner_model.potential_family_evidence_powers(
                        connection,
                        learner_id=session["learner_id"],
                        family_ids={
                            question.family_id for question in eligible
                        },
                        now=now,
                    )
                )
            potential_family_powers = {
                family_id: result.power
                for family_id, result in potential.items()
            }

        scores = [
            self._score(
                question,
                session=session,
                phase=phase,
                prerequisite_distances=prerequisite_distances,
                concepts=concepts,
                stored_states=stored_states,
                objective_states=objective_states,
                projected_objective_states=projected_objective_states,
                beliefs=beliefs,
                exposure=exposure,
                recent_families=recent,
                last_primary_concept=last_primary_concept,
                topic_by_concept=topic_by_concept,
                base_scope=base_scope,
                connected_pairs=connected_pairs,
                readiness=readiness,
                now=now,
                potential_family_powers=potential_family_powers,
                coverage=hybrid_coverage_by_question[question.id],
            )
            for question in eligible
        ]
        scores.sort(key=lambda score: (-score.total, score.question_id))
        if not scores:
            if exploring:
                raise _ExplorationUnserviceable(
                    "no_scored_candidate", exploration_topic_ids
                )
            raise ExhaustedError("Corpus gap: the policy found no eligible question.")

        top_k = scores[
            : min(CANDIDATE_SAMPLING_FRONTIER_LIMIT, len(scores))
        ]
        chosen_score, propensity = self._sample_top_k(
            top_k, seed=session["rng_seed"], step=session["step"]
        )
        question_by_id = {question.id: question for question in eligible}
        chosen = question_by_id[chosen_score.question_id]
        option_ids = [option.id for option in chosen.options]
        order_rng = random.Random(f"options:{session['rng_seed']}:{session['step']}:{chosen.id}")
        order_rng.shuffle(option_ids)

        digest_material = "|".join(
            (
                f"{score.question_id}:{score.total:.8f}:"
                f"{score.coverage_raw_exposures}:"
                f"{score.coverage_diagnostic_information:.12f}:"
                f"{score.coverage_successful_retrieval_families}"
            )
            for score in scores
        )
        candidate_digest = hashlib.sha256(digest_material.encode("utf-8")).hexdigest()
        rationale = self._rationale(
            chosen,
            chosen_score,
            phase,
            focus_concept,
            focus_misconception,
            exploration_topic_ids=exploration_topic_ids if exploring else (),
            exploration_fallback=exploration_fallback,
            fair_coverage_exposure=fair_coverage_exposure,
            persistent_gap_revisit=(
                persistent_gap_revisit_details.get(chosen.objective_id)
                if chosen.objective_id is not None
                else None
            ),
            active_misconception_revisit=active_misconception_revisit,
        )
        decision_id = new_id("dec")

        with self.database.transaction() as connection:
            current = connection.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            current_learner = connection.execute(
                "SELECT revision FROM learners WHERE id = ?", (session["learner_id"],)
            ).fetchone()
            # Candidate scoring and randomized top-k sampling happen outside
            # the writer lock. A safety migration can quarantine historical
            # evidence without changing learner/session revisions, so close
            # that race before any decision or event is written.
            self.database.require_learner_evidence_safe(
                session["learner_id"],
                connection,
            )
            # Scoring runs outside the writer transaction. A performance task
            # can therefore start after the early reconciliation check without
            # changing either session or learner revision. Recheck under this
            # final serialized write boundary so a session can never expose a
            # pending question and an open productive task at the same time.
            open_performance_attempt = connection.execute(
                """SELECT attempt.id
                   FROM performance_attempts attempt
                   WHERE attempt.session_id = ?
                     AND NOT EXISTS (
                         SELECT 1 FROM performance_actions terminal
                         WHERE terminal.attempt_id = attempt.id
                           AND terminal.action_type IN ('submitted', 'abandoned')
                     )
                   ORDER BY attempt.started_at, attempt.id LIMIT 1""",
                (session_id,),
            ).fetchone()
            if open_performance_attempt is not None:
                raise ConflictError(
                    "Session has an active performance task "
                    f"{open_performance_attempt['id']}; submit or abandon it "
                    "before selecting another question."
                )
            release_question = connection.execute(
                """SELECT rq.*, q.version, q.content_hash FROM release_questions rq
                   JOIN questions q ON q.id = rq.question_id
                   WHERE rq.release_id = ? AND rq.question_id = ?""",
                (release_id, chosen.id),
            ).fetchone()
            if (
                not current
                or current["status"] != "active"
                or current["revision"] != session["revision"]
                or current["phase"] != session["phase"]
                or not current_learner
                or current_learner["revision"] != learner_revision
                or current["corpus_release_id"] != release_id
                or not release_question
                or release_question["status"]
                not in {"approved", "calibrated"}
                or connection.execute(
                    "SELECT 1 FROM question_revocations WHERE question_id = ?",
                    (chosen.id,),
                ).fetchone()
            ):
                raise _RetrySelection()
            existing = connection.execute(
                """SELECT id FROM decisions WHERE session_id = ?
                   AND consumed_at IS NULL AND invalidated_at IS NULL""",
                (session_id,),
            ).fetchone()
            if existing:
                # Another caller won the race. Retry through the pending-choice
                # reconciliation above so a cross-model winner is never served
                # without checking its immutable selection boundary.
                raise _RetrySelection()
            else:
                selection_objective_aware = bool(
                    chosen.objective_id or focus_objective
                )
                connection.execute(
                    """INSERT INTO decisions(
                           id, session_id, learner_id, question_id,
                           question_objective_id,
                           question_version, question_content_hash, question_status,
                           evidence_weight, corpus_release_id, session_revision,
                           learner_revision, phase, focus_concept_id,
                           focus_misconception_id, focus_objective_id,
                           pedagogical_role, focus_valid,
                           policy_version, candidate_count, candidate_digest,
                           top_candidates_json, selected_score_json, propensity,
                           option_order_json, rationale, created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        decision_id,
                        session_id,
                        session["learner_id"],
                        chosen.id,
                        chosen.objective_id,
                        release_question["version"],
                        release_question["content_hash"],
                        release_question["status"],
                        release_question["evidence_weight"],
                        release_id,
                        session["revision"],
                        learner_revision,
                        phase.value,
                        focus_concept,
                        focus_misconception,
                        focus_objective,
                        pedagogical_role,
                        int(focus_valid),
                        POLICY_VERSION,
                        len(scores),
                        candidate_digest,
                        json.dumps(
                            [
                                {"question_id": score.question_id, **score.terms()}
                                for score in scores[
                                    :CANDIDATE_AUDIT_PREFIX_LIMIT
                                ]
                            ],
                            sort_keys=True,
                        ),
                        json.dumps(chosen_score.terms(), sort_keys=True),
                        propensity,
                        json.dumps(option_ids),
                        rationale,
                        now.isoformat(),
                    ),
                )
                selection_payload = {
                    "decision_id": decision_id,
                    "question_id": chosen.id,
                    "phase": phase.value,
                    "candidate_count": len(scores),
                    "candidate_digest": candidate_digest,
                    "propensity": propensity,
                    "score": chosen_score.terms(),
                    "option_order": option_ids,
                    "question_version": release_question["version"],
                    "question_content_hash": release_question["content_hash"],
                    "question_status": release_question["status"],
                    "evidence_weight": release_question["evidence_weight"],
                    "corpus_release_id": release_id,
                    "session_revision": session["revision"],
                    "learner_revision": learner_revision,
                    "focus_concept_id": focus_concept,
                    "focus_misconception_id": focus_misconception,
                    "pedagogical_role": pedagogical_role,
                    "focus_valid": focus_valid,
                }
                if selection_objective_aware:
                    selection_payload.update(
                        {
                            "focus_objective_id": focus_objective,
                            "question_objective_id": chosen.objective_id,
                        }
                    )
                self.database.append_event(
                    connection,
                    stream_id=f"learner:{session['learner_id']}",
                    event_type="QuestionSelected",
                    schema_version=question_selected_schema_for(
                        self.learner_model.model_version,
                        objective_aware=selection_objective_aware,
                    ),
                    payload=selection_payload,
                    metadata={
                        "policy_version": POLICY_VERSION,
                        "learner_model_version": self.learner_model.model_version,
                        "corpus_release_id": release_id,
                    },
                    learner_id=session["learner_id"],
                    session_id=session_id,
                    occurred_at=now,
                )
                updated = connection.execute(
                    """UPDATE sessions SET step = step + 1, revision = revision + 1,
                           updated_at = ? WHERE id = ? AND revision = ? AND phase = ?
                           AND status = 'active'""",
                    (now.isoformat(), session_id, session["revision"], session["phase"]),
                )
                if updated.rowcount != 1:
                    raise _RetrySelection()
        durable = self.database.pending_presentation(session_id)
        if not durable:
            raise ExhaustedError("Selection was not persisted.")
        if (
            durable.question.objective_id is not None
            and self.learner_model.model_version not in OBJECTIVE_MODEL_VERSIONS
        ):
            raise ValidationError(
                f"Learner model {self.learner_model.model_version} cannot serve "
                "an objective-aware question; use the current learner model."
            )
        return durable

    def _should_explore(
        self,
        session: dict,
        phase: SessionPhase,
        recent_performance: list[dict],
    ) -> bool:
        """Gate bounded, explicit probes outside a learner's requested topic."""
        if (
            not session.get("topic_id")
            or session.get("exploration_mode") != "adaptive"
            or phase not in {
                SessionPhase.LEARN,
                SessionPhase.DIAGNOSE,
                SessionPhase.REVIEW,
            }
            or session.get("focus_concept_id")
            or session.get("focus_misconception_id")
            or session.get("focus_objective_id")
            or len(recent_performance) < 3
            or session["step"] < 3
            or (session["step"] - 3) % 5 != 0
        ):
            return False
        def supports_exploration(attempt: dict) -> bool:
            if (
                not attempt["correct"]
                or attempt["pedagogical_role"]
                not in {"main", "exploration_probe"}
            ):
                return False
            response_class = classify_response_for_model(
                model_version=attempt.get("learner_model_version"),
                correct=True,
                selected_option_id=attempt.get("selected_option_id"),
                selected_misconception_id=None,
                confidence=attempt["confidence"],
                response_ms=attempt["response_ms"],
                hint_count=attempt["hint_count"],
            )
            return bool(
                response_class.certifies_retrieval
                and (
                    attempt["confidence"] is None
                    or attempt["confidence"] >= 0.65
                )
            )

        return all(
            supports_exploration(attempt)
            for attempt in recent_performance
        )

    @staticmethod
    def _least_exposed_families_by_primary(
        questions: Iterable[Question], exposure: dict
    ) -> list[Question]:
        candidates = list(questions)
        least_by_concept: dict[str, int] = {}
        for question in candidates:
            family_count = exposure["families"].get(
                question.family_id, {}
            ).get("count", 0)
            previous = least_by_concept.get(question.primary_concept_id)
            if previous is None or family_count < previous:
                least_by_concept[question.primary_concept_id] = family_count
        return [
            question
            for question in candidates
            if exposure["families"].get(question.family_id, {}).get("count", 0)
            == least_by_concept[question.primary_concept_id]
        ]

    def _score(
        self,
        question: Question,
        *,
        session: dict,
        phase: SessionPhase,
        prerequisite_distances,
        concepts,
        stored_states,
        objective_states,
        projected_objective_states=None,
        beliefs,
        exposure,
        recent_families: list[str],
        last_primary_concept: str | None,
        topic_by_concept: dict[str, str],
        base_scope: set[str],
        connected_pairs: set[frozenset[str]],
        readiness,
        now: datetime,
        potential_family_powers: Mapping[str, float] | None = None,
        coverage: _HybridCoverage | None = None,
    ) -> CandidateScore:
        states = self.learner_model.states_for_question(
            session["learner_id"], question, concepts, stored_states, now
        )
        objective_state = None
        if question.objective is not None:
            if projected_objective_states is not None:
                objective_state = projected_objective_states.get(
                    question.objective.id
                )
                if objective_state is None:
                    raise ValidationError(
                        "Objective projection cache is incomplete for "
                        f"{question.objective.id}."
                    )
            else:
                objective_state = objective_states.get(question.objective.id)
                if objective_state is None:
                    objective_state = self.learner_model.initial_objective_state(
                        session["learner_id"], question.objective
                    )
                objective_state = self.learner_model.project_objective_state(
                    objective_state, question.objective, now
                )
        predicted = self.learner_model.predict_correct(
            question, states, objective_state=objective_state
        )
        family_exposure = exposure["families"].get(
            question.family_id, {}
        ).get("count", 0)
        if (
            self.learner_model.model_version
            in SPACING_AWARE_FAMILY_MODEL_VERSIONS
        ):
            if potential_family_powers is None:
                # Compatibility for isolated private-score tests. Production
                # selection always supplies the immutable history-derived map.
                potential_family_power = (
                    self.learner_model.family_dependence_discount(
                        family_exposure
                    )
                )
            else:
                potential_family_power = potential_family_powers.get(
                    question.family_id
                )
                if (
                    isinstance(potential_family_power, bool)
                    or not isinstance(
                        potential_family_power, (int, float)
                    )
                    or not 0.0 <= float(potential_family_power) <= 1.0
                ):
                    raise ValidationError(
                        "Planned family evidence power is missing or invalid "
                        f"for {question.family_id}."
                    )
            anticipated_evidence_power = (
                question.status.evidence_weight
                * float(potential_family_power)
            )
            raw_ig = self.learner_model.expected_information_gain(
                question,
                states,
                objective_state=objective_state,
                evidence_power_override=anticipated_evidence_power,
            )
        else:
            # The v5 Gaussian policy remains byte-for-byte compatible with its
            # historical unweighted variance-reduction score.
            raw_ig = self.learner_model.expected_information_gain(
                question, states, objective_state=objective_state
            )
        evidence_weights = self.learner_model.evidence_weights(question)
        information_gain = 1.0 - math.exp(-2.5 * raw_ig)
        target = _TARGET_SUCCESS[phase]
        learning_fit = math.exp(-((predicted - target) / 0.24) ** 2)
        if objective_state is not None:
            concept_need = 1.0 - objective_state.mastery_probability
        else:
            concept_need = sum(
                weight * (1.0 - states[concept_id].mastery_probability)
                for concept_id, weight in evidence_weights.items()
            )

        misconception_value = 0.0
        if session["focus_misconception_id"] in question.misconception_ids:
            misconception_value = 1.0
        elif question.misconception_ids:
            misconception_value = max(
                (beliefs[mid].probability if mid in beliefs else 0.10)
                for mid in question.misconception_ids
            )

        distance = prerequisite_distances.get(question.primary_concept_id)
        if distance is None or distance == 0:
            prerequisite_value = 0.0
        else:
            prerequisite_value = min(1.0, concept_need * (0.55 + 0.45 / distance))

        if objective_state is not None:
            review_value = self.learner_model.retention_due_value(
                objective_state, now
            )
        else:
            review_value = sum(
                weight
                * self.learner_model.retention_due_value(
                    states[concept_id], now
                )
                for concept_id, weight in evidence_weights.items()
            )
        q_exposure = exposure["questions"].get(question.id, 0)
        novelty = 1.0 / (1.0 + 0.55 * q_exposure + 0.25 * family_exposure)
        if question.family_id in recent_families[-4:]:
            novelty *= 0.08

        kind_fit = _KIND_FIT[phase].get(question.kind.value, 0.55)
        primary = question.primary_concept_id
        primary_readiness = readiness[primary]
        if primary_readiness.bottleneck_concept_id is None:
            boundary_fit = 1.0
        else:
            target_support = {
                SessionPhase.DIAGNOSE: 0.55,
                SessionPhase.LEARN: 0.68,
                SessionPhase.REMEDIATE: 0.76,
                SessionPhase.VERIFY: 0.74,
                SessionPhase.REVIEW: 0.82,
            }[phase]
            # Readiness is a floor, not a narrow target: strong prerequisite
            # evidence must never make an otherwise useful advanced item less
            # eligible. Unsupported items are reduced smoothly instead.
            boundary_fit = min(
                1.0, primary_readiness.prerequisite_support / target_support
            )
        if last_primary_concept is None:
            continuity = 0.50
        elif primary == last_primary_concept:
            continuity = 1.0
        elif frozenset((primary, last_primary_concept)) in connected_pairs:
            continuity = 0.85
        elif (
            topic_by_concept.get(primary) is not None
            and topic_by_concept.get(primary)
            == topic_by_concept.get(last_primary_concept)
        ):
            continuity = 0.70
        elif primary in base_scope:
            continuity = 0.40
        else:
            continuity = 0.15
        values = {
            "information_gain": information_gain,
            "learning_fit": learning_fit,
            "concept_need": concept_need,
            "misconception_value": misconception_value,
            "prerequisite_value": prerequisite_value,
            "review_value": review_value,
            "novelty": novelty,
            "kind_fit": kind_fit,
            "boundary_fit": boundary_fit,
        }
        total = sum(_PHASE_WEIGHTS[phase][name] * value for name, value in values.items())
        if session.get("topic_id"):
            total += {
                SessionPhase.DIAGNOSE: 0.05,
                SessionPhase.LEARN: 0.10,
                SessionPhase.REMEDIATE: 0.12,
                SessionPhase.VERIFY: 0.12,
                SessionPhase.REVIEW: 0.08,
            }[phase] * continuity
        if session["focus_concept_id"] and any(
            mapping.concept_id == session["focus_concept_id"] for mapping in question.concepts
        ):
            total += 0.12
        if (
            session.get("focus_objective_id")
            and question.objective_id == session["focus_objective_id"]
        ):
            total += 0.18
        if question.status.value == "calibrated":
            total += 0.03
        return CandidateScore(
            question_id=question.id,
            total=total,
            predicted_correct=predicted,
            information_gain=information_gain,
            learning_fit=learning_fit,
            concept_need=concept_need,
            misconception_value=misconception_value,
            prerequisite_value=prerequisite_value,
            review_value=review_value,
            novelty=novelty,
            kind_fit=kind_fit,
            continuity=continuity,
            boundary_fit=boundary_fit,
            coverage_raw_exposures=(
                coverage.raw_exposures if coverage is not None else 0
            ),
            coverage_diagnostic_information=(
                coverage.diagnostic_information
                if coverage is not None
                else 0.0
            ),
            coverage_successful_retrieval_families=(
                coverage.successful_retrieval_families
                if coverage is not None
                else 0
            ),
        )

    @staticmethod
    def _sample_top_k(
        scores: Iterable[CandidateScore], *, seed: int, step: int
    ) -> tuple[CandidateScore, float]:
        candidates = list(scores)
        temperature = 0.10
        peak = max(score.total for score in candidates)
        weights = [math.exp((score.total - peak) / temperature) for score in candidates]
        total_weight = sum(weights)
        probabilities = [weight / total_weight for weight in weights]
        rng = random.Random(f"policy:{seed}:{step}")
        threshold = rng.random()
        cumulative = 0.0
        for score, probability in zip(candidates, probabilities, strict=True):
            cumulative += probability
            if threshold <= cumulative:
                return score, probability
        return candidates[-1], probabilities[-1]

    @staticmethod
    def _rationale(
        question: Question,
        score: CandidateScore,
        phase: SessionPhase,
        focus_concept: str | None,
        focus_misconception: str | None,
        *,
        exploration_topic_ids: tuple[str, ...] = (),
        exploration_fallback: tuple[str, tuple[str, ...]] | None = None,
        fair_coverage_exposure: int | None = None,
        persistent_gap_revisit: Mapping[str, object] | None = None,
        active_misconception_revisit: str | None = None,
    ) -> str:
        reasons = [
            f"phase={phase.value}",
            f"predicted_success={score.predicted_correct:.2f}",
            f"information={score.information_gain:.2f}",
            f"need={score.concept_need:.2f}",
            "coverage_raw_exposures="
            + str(score.coverage_raw_exposures),
            "coverage_diagnostic_information="
            + format(score.coverage_diagnostic_information, ".12f"),
            "coverage_successful_retrieval_families="
            + str(score.coverage_successful_retrieval_families),
        ]
        if focus_misconception and focus_misconception in question.misconception_ids:
            reasons.append(f"discriminates_misconception={focus_misconception}")
        if question.objective_id:
            reasons.append(f"learning_objective={question.objective_id}")
        elif focus_concept and question.primary_concept_id == focus_concept:
            reasons.append(f"tests_focus={focus_concept}")
        if score.review_value > 0.2:
            reasons.append(f"review_due={score.review_value:.2f}")
        if score.boundary_fit < 0.70:
            reasons.append(f"boundary_fit={score.boundary_fit:.2f}")
        if exploration_topic_ids:
            reasons.append(
                "deliberate_related_topic_probe=" + ",".join(exploration_topic_ids)
            )
        elif exploration_fallback is not None:
            fallback_reason, fallback_topic_ids = exploration_fallback
            reasons.append(
                "exploration_unserviceable="
                + fallback_reason
                + ":"
                + ",".join(fallback_topic_ids)
            )
        elif score.continuity >= 0.70:
            reasons.append(f"continuity={score.continuity:.2f}")
        if fair_coverage_exposure is not None:
            reasons.append(
                f"fair_coverage_target_exposures={fair_coverage_exposure}"
            )
        if persistent_gap_revisit is not None:
            due_at = persistent_gap_revisit["next_review_at"]
            if (
                not isinstance(due_at, datetime)
                or due_at.tzinfo is None
                or due_at.utcoffset() is None
            ):
                raise ValidationError(
                    "Persistent-gap rationale lacks a valid due timestamp."
                )
            reasons.extend(
                (
                    f"persistent_gap_revisit={question.objective_id}",
                    "persistent_gap_observed_families="
                    + str(
                        persistent_gap_revisit[
                            "observed_response_families"
                        ]
                    ),
                    "persistent_gap_mastery="
                    + format(
                        persistent_gap_revisit["mastery_probability"],
                        ".6f",
                    ),
                    "persistent_gap_cold_prior="
                    + format(
                        persistent_gap_revisit[
                            "cold_start_mastery_probability"
                        ],
                        ".6f",
                    ),
                    "persistent_gap_due_at="
                    + due_at.astimezone(timezone.utc).isoformat(),
                    "persistent_gap_episode_spend="
                    + str(persistent_gap_revisit["episode_spend"]),
                    "persistent_gap_episode_budget="
                    + str(persistent_gap_revisit["episode_budget"]),
                )
            )
        if active_misconception_revisit is not None:
            reasons.append(
                "active_misconception_revisit="
                + active_misconception_revisit
            )
        rationale = "; ".join(reasons)
        if persistent_gap_revisit is not None:
            AdaptivePolicy._persistent_gap_marker(
                rationale=rationale,
                policy_version=POLICY_VERSION,
                decision_objective_id=question.objective_id,
            )
        return rationale
