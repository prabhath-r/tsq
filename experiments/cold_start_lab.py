#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0

"""Measure TSQ cold-start difficulty without changing the adaptive policy.

This laboratory drives the production engine on disposable databases for
credible-correct, credible-wrong, and explicit-abstention learners.  It records
what the policy actually serves, including authored difficulty, predicted
success, objective depth, focused transitions, and categorized gaps.  The
artifact is diagnostic evidence about the current corpus and selector; it is
not learner calibration or proof of instructional efficacy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tsq.corpus import read_and_parse  # noqa: E402
from tsq.engine import AdaptiveEngine  # noqa: E402
from tsq.models import Presentation  # noqa: E402
from tsq.objective_posterior import decode_objective_posterior  # noqa: E402
from tsq.policy import (  # noqa: E402
    CANDIDATE_AUDIT_PREFIX_LIMIT,
    CANDIDATE_SAMPLING_FRONTIER_LIMIT,
    POLICY_VERSION,
)
from tsq.simulation import (  # noqa: E402
    BehavioralSimulator,
    SIMULATION_FEEDBACK_PROTOCOL_VERSION,
    SyntheticAnswer,
    SyntheticLearner,
)
from tsq.store import (  # noqa: E402
    Database,
    question_runtime_activation_safe,
)


LAB_VERSION = "cold-start-lab-v3"
SEMANTIC_PROJECTION_SIGNATURE_SCHEMA = 1
DEFAULT_CORPUS = PROJECT_ROOT / "corpus" / "ai_curriculum.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "results" / "cold_start_lab.json"
DEFAULT_TOPICS = (
    "t_large_language_models",
    "t_transformers",
    "t_retrieval_augmented_generation",
    "t_llm_agents",
)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_STEPS = 9
START = datetime(2110, 1, 5, 9, 0, tzinfo=timezone.utc)
# A fixed descriptive reference for comparing artifacts. It intentionally is
# not imported from the policy's private phase-scoring implementation.
LEARN_PHASE_REFERENCE_SUCCESS = 0.68
INTRODUCTORY_DIFFICULTY_CEILING = -0.50
ADVANCED_DIFFICULTY_FLOOR = 0.75
CANDIDATE_PREFIX_LIMIT = CANDIDATE_AUDIT_PREFIX_LIMIT
SAMPLING_FRONTIER_LIMIT = CANDIDATE_SAMPLING_FRONTIER_LIMIT
CANDIDATE_SCORE_TERM_KEYS = frozenset(
    {
        "total",
        "predicted_correct",
        "information_gain",
        "learning_fit",
        "concept_need",
        "misconception_value",
        "prerequisite_value",
        "review_value",
        "novelty",
        "kind_fit",
        "continuity",
        "boundary_fit",
        "coverage_raw_exposures",
        "coverage_diagnostic_information",
        "coverage_successful_retrieval_families",
    }
)
CANDIDATE_INTEGER_TERM_KEYS = frozenset(
    {
        "coverage_raw_exposures",
        "coverage_successful_retrieval_families",
    }
)


class ColdStartInvariantError(RuntimeError):
    """A laboratory integrity or determinism invariant failed."""


@dataclass(frozen=True, slots=True)
class AbstainingLearner:
    """An explicit abstention with no undefined confidence value."""

    name: str = "abstaining"
    rule: str = "explicit-abstention-no-confidence"
    response_model: str = "deterministic-abstention"
    response_ms: int = 4_000

    def answer(
        self,
        presentation: Presentation,
        *,
        simulation_seed: int,
        trial_index: int,
        encounter: int,
    ) -> SyntheticAnswer:
        del presentation, simulation_seed, trial_index, encounter
        return SyntheticAnswer(
            selected_option_id=None,
            correct=False,
            ground_truth_probability=0.50,
            confidence=None,
            response_ms=self.response_ms,
            hint_count=0,
        )


def _profiles() -> tuple[SyntheticLearner | AbstainingLearner, ...]:
    return (
        SyntheticLearner(
            "credible-correct",
            forced_correctness=True,
            confidence_override=0.90,
            base_response_ms=4_000,
            slip_probability=0.0,
            guess_probability=0.0,
            seed=101,
        ),
        SyntheticLearner(
            "credible-wrong",
            forced_correctness=False,
            confidence_override=0.90,
            base_response_ms=4_000,
            slip_probability=0.0,
            guess_probability=0.0,
            seed=202,
        ),
        AbstainingLearner(),
    )


def canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _semantic_projection_signature(
    database: Database,
    learner_id: str,
) -> dict[str, Any]:
    """Hash stable learner semantics while excluding random event identities."""

    with database.read() as connection:
        learner = connection.execute(
            "SELECT id, revision FROM learners WHERE id = ?",
            (learner_id,),
        ).fetchone()
        if learner is None:
            raise ColdStartInvariantError(
                f"Learner {learner_id} disappeared before projection hashing."
            )
        skill_states = [
            dict(row)
            for row in connection.execute(
                """SELECT learner_id, concept_id, mean, variance,
                          stability_hours, exposures, last_seen_at,
                          next_review_at, evidence_mass, model_version
                   FROM skill_states WHERE learner_id = ?
                   ORDER BY concept_id""",
                (learner_id,),
            )
        ]
        objective_states = [
            dict(row)
            for row in connection.execute(
                """SELECT learner_id, objective_id, mean, variance,
                          stability_hours, exposures, last_seen_at,
                          next_review_at, evidence_mass, model_version
                   FROM objective_states WHERE learner_id = ?
                   ORDER BY objective_id""",
                (learner_id,),
            )
        ]
        misconception_beliefs = [
            dict(row)
            for row in connection.execute(
                """SELECT learner_id, misconception_id, log_odds,
                          evidence_count, last_seen_at, model_version
                   FROM misconception_beliefs WHERE learner_id = ?
                   ORDER BY misconception_id""",
                (learner_id,),
            )
        ]
        skill_families = [
            dict(row)
            for row in connection.execute(
                """SELECT learner_id, concept_id, family_id, kind,
                          first_unguided_correct_at,
                          last_unguided_correct_at,
                          delayed_unguided_correct_at
                   FROM learner_skill_families WHERE learner_id = ?
                   ORDER BY concept_id, family_id""",
                (learner_id,),
            )
        ]
        objective_families = [
            dict(row)
            for row in connection.execute(
                """SELECT learner_id, objective_id, family_id, kind,
                          first_unguided_correct_at,
                          last_unguided_correct_at,
                          delayed_unguided_correct_at
                   FROM learner_objective_families WHERE learner_id = ?
                   ORDER BY objective_id, family_id""",
                (learner_id,),
            )
        ]
        grid_rows = connection.execute(
            """SELECT learner_id, objective_id, posterior_schema_version,
                      algorithm, grid_id, codec, posterior_blob,
                      posterior_sha256, mean, variance, mastery_probability,
                      expected_competence, edge_mass,
                      mastery_probability_error_bound, evidence_mass,
                      acquisition_mass, model_version
               FROM objective_grid_states WHERE learner_id = ?
               ORDER BY objective_id""",
            (learner_id,),
        ).fetchall()

    objective_grid_states: list[dict[str, Any]] = []
    for row in grid_rows:
        posterior = decode_objective_posterior(
            bytes(row["posterior_blob"]),
            expected_digest=row["posterior_sha256"],
        )
        pending_observations = []
        for observation in posterior.pending_observations:
            semantic_observation = observation.as_payload()
            semantic_observation.pop("observation_id")
            pending_observations.append(semantic_observation)
        pending_observations.sort(
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        posterior_semantics = {
            "schema_version": row["posterior_schema_version"],
            "algorithm": row["algorithm"],
            "codec": row["codec"],
            "grid_id": row["grid_id"],
            "prior_mastery": posterior.prior_mastery,
            "prior_variance": posterior.prior_variance,
            "anchor_log_density": list(posterior.anchor_log_density),
            "current_log_density": list(posterior.log_density),
            "pending_observations": pending_observations,
            "committed_evidence_mass": posterior.committed_evidence_mass,
            "acquisition_mass": posterior.acquisition_mass,
        }
        objective_grid_states.append(
            {
                "learner_id": row["learner_id"],
                "objective_id": row["objective_id"],
                "posterior_schema_version": row[
                    "posterior_schema_version"
                ],
                "algorithm": row["algorithm"],
                "grid_id": row["grid_id"],
                "codec": row["codec"],
                "mean": row["mean"],
                "variance": row["variance"],
                "mastery_probability": row["mastery_probability"],
                "expected_competence": row["expected_competence"],
                "edge_mass": row["edge_mass"],
                "mastery_probability_error_bound": row[
                    "mastery_probability_error_bound"
                ],
                "evidence_mass": row["evidence_mass"],
                "acquisition_mass": row["acquisition_mass"],
                "model_version": row["model_version"],
                "posterior_semantics_sha256": canonical_hash(
                    posterior_semantics
                ),
            }
        )

    payload = {
        "signature_schema_version": (
            SEMANTIC_PROJECTION_SIGNATURE_SCHEMA
        ),
        "learner_id": learner["id"],
        "learner_revision": learner["revision"],
        "skill_states": skill_states,
        "objective_states": objective_states,
        "objective_grid_states": objective_grid_states,
        "misconception_beliefs": misconception_beliefs,
        "skill_families": skill_families,
        "objective_families": objective_families,
    }
    return {
        "schema_version": SEMANTIC_PROJECTION_SIGNATURE_SCHEMA,
        "sha256": canonical_hash(payload),
        "excluded_random_provenance": [
            "projection as_of_event_id values",
            "posterior pending observation IDs",
            "stored posterior byte digest",
        ],
    }


def _objective_depths(bundle: Mapping[str, Any]) -> dict[str, int]:
    objective_ids = {
        row["id"] for row in bundle.get("learning_objectives", ())
    }
    prerequisites: dict[str, set[str]] = {
        objective_id: set() for objective_id in objective_ids
    }
    for edge in bundle.get("objective_edges", ()):
        source = edge["source"]
        target = edge["target"]
        if source not in objective_ids or target not in objective_ids:
            raise ColdStartInvariantError(
                "The parsed objective graph references an unknown objective."
            )
        prerequisites[target].add(source)

    depths: dict[str, int] = {}
    visiting: set[str] = set()

    def depth(objective_id: str) -> int:
        if objective_id in depths:
            return depths[objective_id]
        if objective_id in visiting:
            raise ColdStartInvariantError(
                "The objective prerequisite graph contains a cycle."
            )
        visiting.add(objective_id)
        parents = prerequisites[objective_id]
        value = 0 if not parents else 1 + max(depth(parent) for parent in parents)
        visiting.remove(objective_id)
        depths[objective_id] = value
        return value

    for objective_id in sorted(objective_ids):
        depth(objective_id)
    return depths


def _topic_descendants(
    topics: Sequence[Mapping[str, Any]], topic_id: str
) -> set[str]:
    children: dict[str, set[str]] = defaultdict(set)
    known = {row["id"] for row in topics}
    if topic_id not in known:
        raise ValueError(f"Unknown topic {topic_id}.")
    for row in topics:
        parent = row.get("parent_id")
        if parent:
            children[parent].add(row["id"])
    result: set[str] = set()
    pending = [topic_id]
    while pending:
        current = pending.pop()
        if current in result:
            continue
        result.add(current)
        pending.extend(sorted(children[current], reverse=True))
    return result


def _primary_concept(question: Mapping[str, Any]) -> str:
    primary = [
        mapping["concept_id"]
        for mapping in question["concepts"]
        if mapping["role"] == "primary"
    ]
    if len(primary) != 1:
        raise ColdStartInvariantError(
            f"Question {question.get('id')} lacks one primary concept."
        )
    return primary[0]


def _corpus_topic_summary(
    bundle: Mapping[str, Any], topic_id: str
) -> dict[str, Any]:
    topics = bundle["topics"]
    descendants = _topic_descendants(topics, topic_id)
    concept_ids = {
        concept_id
        for topic in topics
        if topic["id"] in descendants
        for concept_id in topic.get("concept_ids", ())
    }
    questions = [
        question
        for question in bundle["questions"]
        if question["status"] in {"approved", "calibrated"}
        and _primary_concept(question) in concept_ids
    ]
    difficulties = [float(question["difficulty"]) for question in questions]
    return {
        "topic_id": topic_id,
        "approved_or_calibrated_questions": len(questions),
        "authored_primary_scope_contract": (
            "The static summary uses the authored primary concept in this "
            "topic or a descendant. Runtime scope remains authoritative."
        ),
        "difficulty": {
            "minimum": min(difficulties) if difficulties else None,
            "median": statistics.median(difficulties) if difficulties else None,
            "mean": statistics.fmean(difficulties) if difficulties else None,
            "maximum": max(difficulties) if difficulties else None,
            "introductory_count": sum(
                value < INTRODUCTORY_DIFFICULTY_CEILING
                for value in difficulties
            ),
            "advanced_count": sum(
                value > ADVANCED_DIFFICULTY_FLOOR for value in difficulties
            ),
        },
    }


def _transition_reasons(
    database: Database, learner_id: str
) -> Counter[str]:
    reasons: Counter[str] = Counter()
    with database.read() as connection:
        rows = connection.execute(
            """SELECT outcome_json FROM attempts
               WHERE learner_id = ? ORDER BY answered_at, id""",
            (learner_id,),
        ).fetchall()
    for row in rows:
        try:
            outcome = json.loads(row["outcome_json"])
        except (TypeError, ValueError) as exc:
            raise ColdStartInvariantError(
                f"Learner {learner_id} has an invalid attempt outcome."
            ) from exc
        reason = outcome.get("transition_reason")
        if not isinstance(reason, str) or not reason:
            raise ColdStartInvariantError(
                f"Learner {learner_id} has no committed transition reason."
            )
        reasons[reason] += 1
    return reasons


def _validated_score_terms(
    value: object, *, label: str
) -> dict[str, float | int]:
    if not isinstance(value, dict) or set(value) != CANDIDATE_SCORE_TERM_KEYS:
        raise ColdStartInvariantError(
            f"{label} does not contain the exact candidate-score terms."
        )
    result: dict[str, float | int] = {}
    for key in sorted(CANDIDATE_SCORE_TERM_KEYS):
        item = value[key]
        if key in CANDIDATE_INTEGER_TERM_KEYS:
            if type(item) is not int or item < 0:
                raise ColdStartInvariantError(
                    f"{label} term {key} must be a non-negative integer."
                )
        elif (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise ColdStartInvariantError(
                f"{label} term {key} must be a finite number."
            )
        result[key] = item
    if (
        result["coverage_successful_retrieval_families"]
        > result["coverage_raw_exposures"]
    ):
        raise ColdStartInvariantError(
            f"{label} has more successful families than raw exposures."
        )
    if result["coverage_diagnostic_information"] < 0:
        raise ColdStartInvariantError(
            f"{label} has negative diagnostic information."
        )
    return result


def _ranked_candidate_digest(
    ranked: Sequence[tuple[str, Mapping[str, float | int]]],
) -> str:
    material = "|".join(
        (
            f"{question_id}:{float(score['total']):.8f}:"
            f"{score['coverage_raw_exposures']}:"
            f"{float(score['coverage_diagnostic_information']):.12f}:"
            f"{score['coverage_successful_retrieval_families']}"
        )
        for question_id, score in ranked
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _candidate_inventory_from_row(
    *,
    row: Mapping[str, Any],
    step: Any,
    question_metadata: Mapping[str, Mapping[str, Any]],
    trace_label: str,
) -> dict[str, Any]:
    expected_question_id = step.question_id
    expected_release_id = row["corpus_release_id"]
    selected_evidence_weight = row["selected_evidence_weight"]
    if (
        row["attempted_question_id"] != expected_question_id
        or row["selected_question_id"] != expected_question_id
        or row["selected_family_id"] != step.family_id
        or row["selected_question_kind"] != step.question_kind
        or row["selected_objective_id"]
        != step.learning_objective_id
        or row["phase"] != step.phase_before.value
        or row["pedagogical_role"] != step.pedagogical_role
        or row["selected_at"] != step.selected_at.isoformat()
        or row["answered_at"] != step.answered_at.isoformat()
        or row["selected_question_status"] not in {
            "approved",
            "calibrated",
        }
        or isinstance(selected_evidence_weight, bool)
        or not isinstance(selected_evidence_weight, (int, float))
        or not math.isfinite(float(selected_evidence_weight))
        or float(selected_evidence_weight) <= 0.0
    ):
        raise ColdStartInvariantError(
            f"{trace_label} does not align with its committed simulation step."
        )

    candidate_count = row["candidate_count"]
    if type(candidate_count) is not int or candidate_count <= 0:
        raise ColdStartInvariantError(
            f"{trace_label} has an invalid candidate count."
        )
    candidate_digest = row["candidate_digest"]
    if (
        not isinstance(candidate_digest, str)
        or len(candidate_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in candidate_digest
        )
    ):
        raise ColdStartInvariantError(
            f"{trace_label} has an invalid candidate digest."
        )
    try:
        top_candidates = json.loads(row["top_candidates_json"])
        selected_score_value = json.loads(row["selected_score_json"])
    except (TypeError, ValueError) as exc:
        raise ColdStartInvariantError(
            f"{trace_label} has invalid candidate JSON."
        ) from exc
    selected_score = _validated_score_terms(
        selected_score_value,
        label=f"{trace_label} selected score",
    )
    if selected_score["predicted_correct"] != step.predicted_correct:
        raise ColdStartInvariantError(
            f"{trace_label} predicted success differs from the simulation step."
        )
    expected_prefix_count = min(CANDIDATE_PREFIX_LIMIT, candidate_count)
    if (
        not isinstance(top_candidates, list)
        or len(top_candidates) != expected_prefix_count
    ):
        raise ColdStartInvariantError(
            f"{trace_label} has an incomplete ranked prefix."
        )

    ranked_for_digest: list[tuple[str, Mapping[str, float | int]]] = []
    ranked_candidates: list[dict[str, Any]] = []
    seen_question_ids: set[str] = set()
    previous_key: tuple[float, str] | None = None
    for rank, candidate in enumerate(top_candidates, start=1):
        if not isinstance(candidate, dict):
            raise ColdStartInvariantError(
                f"{trace_label} contains a non-object candidate."
            )
        question_id = candidate.get("question_id")
        if (
            not isinstance(question_id, str)
            or question_id not in question_metadata
            or question_id in seen_question_ids
        ):
            raise ColdStartInvariantError(
                f"{trace_label} has invalid ranked candidate identity."
            )
        score = _validated_score_terms(
            {
                key: value
                for key, value in candidate.items()
                if key != "question_id"
            },
            label=f"{trace_label} candidate {rank}",
        )
        rank_key = (-float(score["total"]), question_id)
        if previous_key is not None and rank_key < previous_key:
            raise ColdStartInvariantError(
                f"{trace_label} is not stored in production rank order."
            )
        previous_key = rank_key
        seen_question_ids.add(question_id)
        ranked_for_digest.append((question_id, score))
        metadata = question_metadata[question_id]
        required_metadata = {
            "corpus_release_id",
            "family_id",
            "learning_objective_id",
            "primary_concept_id",
            "question_kind",
            "difficulty",
            "global_status",
            "release_status",
            "evidence_weight",
            "revoked",
            "runtime_activation_safe",
        }
        if (
            not isinstance(metadata, Mapping)
            or not required_metadata.issubset(metadata)
        ):
            raise ColdStartInvariantError(
                f"{trace_label} has incomplete candidate metadata."
            )
        evidence_weight = metadata["evidence_weight"]
        if (
            metadata["corpus_release_id"] != expected_release_id
            or metadata["global_status"] not in {
                "approved",
                "calibrated",
            }
            or metadata["release_status"] not in {
                "approved",
                "calibrated",
            }
            or isinstance(evidence_weight, bool)
            or not isinstance(evidence_weight, (int, float))
            or not math.isfinite(float(evidence_weight))
            or float(evidence_weight) <= 0.0
            or metadata["revoked"] is not False
            or metadata["runtime_activation_safe"] is not True
        ):
            raise ColdStartInvariantError(
                f"{trace_label} contains a runtime-ineligible candidate."
            )
        ranked_candidates.append(
            {
                "rank": rank,
                "question_id": question_id,
                "family_id": metadata["family_id"],
                "learning_objective_id": metadata[
                    "learning_objective_id"
                ],
                "primary_concept_id": metadata["primary_concept_id"],
                "question_kind": metadata["question_kind"],
                "difficulty": metadata["difficulty"],
                "score": score,
            }
        )

    selected = [
        candidate
        for candidate in ranked_candidates
        if candidate["question_id"] == expected_question_id
    ]
    if (
        len(selected) != 1
        or selected[0]["rank"]
        > min(SAMPLING_FRONTIER_LIMIT, candidate_count)
    ):
        raise ColdStartInvariantError(
            f"{trace_label} selected outside the production sampling frontier."
        )
    if (
        selected[0]["family_id"] != step.family_id
        or selected[0]["learning_objective_id"]
        != step.learning_objective_id
        or selected[0]["question_kind"] != step.question_kind
        or selected[0]["primary_concept_id"]
        != step.surface_primary_concept_id
    ):
        raise ColdStartInvariantError(
            f"{trace_label} selected corpus metadata differs from its "
            "committed simulation step."
        )
    if selected[0]["score"] != selected_score:
        raise ColdStartInvariantError(
            f"{trace_label} selected score differs from its ranked prefix."
        )

    inventory_complete = candidate_count <= CANDIDATE_PREFIX_LIMIT
    rank_quantized_coverage_digest_verified: bool | None = None
    if inventory_complete:
        if _ranked_candidate_digest(ranked_for_digest) != candidate_digest:
            raise ColdStartInvariantError(
                f"{trace_label} complete rank-and-quantized-coverage digest "
                "does not verify."
            )
        rank_quantized_coverage_digest_verified = True
    selected_difficulty = float(selected[0]["difficulty"])
    prefix_difficulties = [
        float(candidate["difficulty"])
        for candidate in ranked_candidates
    ]
    prefix_difficulty = {
        "minimum": min(prefix_difficulties),
        "maximum": max(prefix_difficulties),
        "introductory_count": sum(
            difficulty < INTRODUCTORY_DIFFICULTY_CEILING
            for difficulty in prefix_difficulties
        ),
        "easier_than_selected_count": sum(
            difficulty < selected_difficulty
            for difficulty in prefix_difficulties
        ),
        "selected_minus_minimum": (
            selected_difficulty - min(prefix_difficulties)
        ),
    }
    return {
        "eligible_scored_candidate_count": candidate_count,
        "stored_ranked_prefix_count": len(ranked_candidates),
        "ranked_inventory_complete": inventory_complete,
        "unobserved_candidate_count": (
            candidate_count - len(ranked_candidates)
        ),
        "sampling_frontier_count": min(
            SAMPLING_FRONTIER_LIMIT, candidate_count
        ),
        "selected_rank": selected[0]["rank"],
        "candidate_digest": candidate_digest,
        "rank_and_quantized_coverage_digest_verified": (
            rank_quantized_coverage_digest_verified
        ),
        "stored_ranked_prefix_sha256": canonical_hash(top_candidates),
        "ranked_prefix_difficulty": prefix_difficulty,
        "complete_inventory_difficulty": (
            prefix_difficulty if inventory_complete else None
        ),
        "ranked_candidates": ranked_candidates,
    }


def _candidate_inventories(
    database: Database,
    learner_id: str,
    steps: Sequence[Any],
    question_metadata: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    with database.read() as connection:
        session_rows = connection.execute(
            """SELECT id FROM sessions
               WHERE learner_id = ? ORDER BY created_at, id""",
            (learner_id,),
        ).fetchall()
        if len(session_rows) != 1:
            raise ColdStartInvariantError(
                f"Learner {learner_id} must have exactly one disposable "
                "cold-start session."
            )
        rows = connection.execute(
            """SELECT attempts.question_id AS attempted_question_id,
                      attempts.answered_at,
                      decisions.question_id AS selected_question_id,
                      question.family_id AS selected_family_id,
                      question.kind AS selected_question_kind,
                      decisions.question_objective_id
                          AS selected_objective_id,
                      decisions.phase,
                      decisions.pedagogical_role,
                      decisions.created_at AS selected_at,
                      decisions.candidate_count,
                      decisions.candidate_digest,
                      decisions.top_candidates_json,
                      decisions.selected_score_json,
                      decisions.corpus_release_id,
                      decisions.question_status
                          AS selected_question_status,
                      decisions.evidence_weight
                          AS selected_evidence_weight
               FROM attempts
               JOIN decisions ON decisions.id = attempts.decision_id
               JOIN questions question
                 ON question.id = decisions.question_id
               WHERE attempts.learner_id = ?
                 AND attempts.session_id = ?
               ORDER BY attempts.answered_at, decisions.created_at,
                        attempts.question_id""",
            (learner_id, session_rows[0]["id"]),
        ).fetchall()
    if len(rows) != len(steps):
        raise ColdStartInvariantError(
            f"Learner {learner_id} has {len(rows)} candidate traces for "
            f"{len(steps)} simulation steps."
        )
    return [
        _candidate_inventory_from_row(
            row=row,
            step=step,
            question_metadata=question_metadata,
            trace_label=f"Learner {learner_id} candidate trace {index}",
        )
        for index, (row, step) in enumerate(
            zip(rows, steps, strict=True),
            start=1,
        )
    ]


def _release_question_metadata(
    database: Database,
) -> dict[str, dict[str, Any]]:
    with database.read() as connection:
        release_id = database.get_active_release_id(connection)
        rows = connection.execute(
            """SELECT question.id, question.family_id,
                      question.kind AS question_kind,
                      question.difficulty,
                      question.status AS global_status,
                      membership.status AS release_status,
                      membership.evidence_weight,
                      primary_mapping.concept_id
                          AS primary_concept_id,
                      objective_mapping.objective_id
                          AS learning_objective_id,
                      revocation.question_id IS NOT NULL AS revoked
               FROM release_questions membership
               JOIN questions question
                 ON question.id = membership.question_id
               JOIN question_concepts primary_mapping
                 ON primary_mapping.question_id = question.id
                AND primary_mapping.role = 'primary'
               LEFT JOIN release_question_objectives objective_mapping
                 ON objective_mapping.release_id = membership.release_id
                AND objective_mapping.question_id = question.id
               LEFT JOIN question_revocations revocation
                 ON revocation.question_id = question.id
               WHERE membership.release_id = ?
               ORDER BY question.id""",
            (release_id,),
        ).fetchall()
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            question = database.get_question(
                row["id"],
                connection,
                release_id=release_id,
            )
            result[row["id"]] = {
                "corpus_release_id": release_id,
                "family_id": row["family_id"],
                "learning_objective_id": row[
                    "learning_objective_id"
                ],
                "primary_concept_id": row["primary_concept_id"],
                "question_kind": row["question_kind"],
                "difficulty": float(row["difficulty"]),
                "global_status": row["global_status"],
                "release_status": row["release_status"],
                "evidence_weight": float(row["evidence_weight"]),
                "revoked": bool(row["revoked"]),
                "runtime_activation_safe": (
                    question_runtime_activation_safe(
                        question,
                        status=row["release_status"],
                    )
                ),
            }
    return result


def _unlabeled_scope_excursions(report: Any) -> dict[str, list[str]]:
    """Separate deliberate exploration from accidental scope leakage."""

    non_exploration_steps = [
        step
        for step in report.steps
        if step.pedagogical_role != "exploration_probe"
    ]
    observed_without_exploration = {
        "concepts": {
            step.evidence_anchor_concept_id
            for step in non_exploration_steps
        },
        "objectives": {
            step.learning_objective_id
            for step in non_exploration_steps
            if step.learning_objective_id is not None
        },
        "questions": {
            step.question_id for step in non_exploration_steps
        },
        "families": {step.family_id for step in non_exploration_steps},
        "evidence_families": {
            (
                f"objective:{step.learning_objective_id}|family:{step.family_id}"
                if step.learning_objective_id is not None
                else (
                    f"concept:{step.evidence_anchor_concept_id}|"
                    f"family:{step.family_id}"
                )
            )
            for step in non_exploration_steps
        },
        "objective_families": {
            f"objective:{step.learning_objective_id}|family:{step.family_id}"
            for step in non_exploration_steps
            if step.learning_objective_id is not None
        },
    }
    outside = {
        "concepts": set(
            report.coverage.observed_outside_scope_concepts
        ),
        "objectives": set(
            report.coverage.observed_outside_scope_objectives
        ),
        "questions": set(
            report.coverage.observed_outside_scope_questions
        ),
        "families": set(
            report.coverage.observed_outside_scope_families
        ),
        "evidence_families": set(
            report.coverage.observed_outside_scope_evidence_families
        ),
        "objective_families": set(
            report.coverage.observed_outside_scope_objective_families
        ),
    }
    return {
        dimension: sorted(
            identifiers & observed_without_exploration[dimension]
        )
        for dimension, identifiers in outside.items()
    }


def _run_payload(
    *,
    report: Any,
    database: Database,
    question_difficulty: Mapping[str, float],
    question_metadata: Mapping[str, Mapping[str, Any]],
    objective_depth: Mapping[str, int],
) -> dict[str, Any]:
    difficulties = [
        question_difficulty[step.question_id] for step in report.steps
    ]
    predicted = [step.predicted_correct for step in report.steps]
    learn_phase_predicted = [
        step.predicted_correct
        for step in report.steps
        if step.phase_before.value == "learn"
    ]
    reasons = _transition_reasons(database, report.learner_id)
    unlabeled_scope_excursions = _unlabeled_scope_excursions(report)
    candidate_inventories = _candidate_inventories(
        database,
        report.learner_id,
        report.steps,
        question_metadata,
    )
    steps = [
        {
            "index": step.index,
            "phase_before": step.phase_before.value,
            "phase_after": step.phase_after.value,
            "pedagogical_role": step.pedagogical_role,
            "question_id": step.question_id,
            "family_id": step.family_id,
            "question_kind": step.question_kind,
            "learning_objective_id": step.learning_objective_id,
            "objective_depth": (
                objective_depth[step.learning_objective_id]
                if step.learning_objective_id is not None
                else None
            ),
            "difficulty": question_difficulty[step.question_id],
            "predicted_success": step.predicted_correct,
            "selected_option_id": step.selected_option_id,
            "correct": step.actual_correct,
            "focus_objective_before": step.focus_objective_before,
            "focus_objective_after": step.focus_objective_after,
            "candidate_inventory": candidate_inventory,
        }
        for step, candidate_inventory in zip(
            report.steps,
            candidate_inventories,
            strict=True,
        )
    ]
    candidate_counts = [
        inventory["eligible_scored_candidate_count"]
        for inventory in candidate_inventories
    ]
    complete_inventory_difficulties = [
        inventory["complete_inventory_difficulty"]
        for inventory in candidate_inventories
        if inventory["complete_inventory_difficulty"] is not None
    ]
    focused_complete_inventory_difficulties = [
        inventory["complete_inventory_difficulty"]
        for step, inventory in zip(
            report.steps,
            candidate_inventories,
            strict=True,
        )
        if step.pedagogical_role != "exploration_probe"
        and inventory["complete_inventory_difficulty"] is not None
    ]
    return {
        "profile": report.profile_name,
        "topic_id": report.root_concept_id,
        "seed": report.policy_seed,
        "behavior_signature": report.behavior_signature(),
        "semantic_projection_signature": (
            _semantic_projection_signature(database, report.learner_id)
        ),
        "attempted": report.attempted,
        "correct": report.correct,
        "abstained": sum(
            step.selected_option_id is None for step in report.steps
        ),
        "difficulty": {
            "first": difficulties[0] if difficulties else None,
            "minimum": min(difficulties) if difficulties else None,
            "mean": statistics.fmean(difficulties) if difficulties else None,
            "maximum": max(difficulties) if difficulties else None,
            "introductory_served": sum(
                value < INTRODUCTORY_DIFFICULTY_CEILING
                for value in difficulties
            ),
            "advanced_served": sum(
                value > ADVANCED_DIFFICULTY_FLOOR for value in difficulties
            ),
        },
        "predicted_success": {
            "first": predicted[0] if predicted else None,
            "mean_all_phases": (
                statistics.fmean(predicted) if predicted else None
            ),
            "learn_phase_count": len(learn_phase_predicted),
            "learn_phase_mean": (
                statistics.fmean(learn_phase_predicted)
                if learn_phase_predicted
                else None
            ),
            "learn_phase_at_or_above_reference": sum(
                value >= LEARN_PHASE_REFERENCE_SUCCESS
                for value in learn_phase_predicted
            ),
            "learn_phase_reference": LEARN_PHASE_REFERENCE_SUCCESS,
        },
        "candidate_inventory": {
            "minimum_eligible_scored_candidates": (
                min(candidate_counts) if candidate_counts else None
            ),
            "mean_eligible_scored_candidates": (
                statistics.fmean(candidate_counts)
                if candidate_counts
                else None
            ),
            "maximum_eligible_scored_candidates": (
                max(candidate_counts) if candidate_counts else None
            ),
            "complete_ranked_inventory_steps": sum(
                inventory["ranked_inventory_complete"]
                for inventory in candidate_inventories
            ),
            "complete_inventory_steps_without_introductory_candidate": sum(
                difficulty["introductory_count"] == 0
                for difficulty in complete_inventory_difficulties
            ),
            "focused_complete_inventory_steps_without_introductory_candidate": sum(
                difficulty["introductory_count"] == 0
                for difficulty in focused_complete_inventory_difficulties
            ),
            "mean_selected_difficulty_above_complete_inventory_minimum": (
                statistics.fmean(
                    difficulty["selected_minus_minimum"]
                    for difficulty in complete_inventory_difficulties
                )
                if complete_inventory_difficulties
                else None
            ),
            "below_full_sampling_frontier_steps": sum(
                count < SAMPLING_FRONTIER_LIMIT
                for count in candidate_counts
            ),
            "singleton_candidate_steps": sum(
                count == 1 for count in candidate_counts
            ),
        },
        "maximum_objective_depth": max(
            (
                objective_depth[step.learning_objective_id]
                for step in report.steps
                if step.learning_objective_id is not None
            ),
            default=None,
        ),
        "transition_reasons": dict(sorted(reasons.items())),
        "prerequisite_descents": reasons["descend_to_evidence_boundary"],
        "exact_repeats": report.exact_repeat_count,
        "family_repeats": report.family_repeat_count,
        "remediation_exact_repeats": report.remediation_exact_repeat_count,
        "remediation_family_repeats": report.remediation_family_repeat_count,
        "deliberate_exploration_questions": sum(
            step.pedagogical_role == "exploration_probe"
            for step in report.steps
        ),
        "outside_scope_questions": len(
            report.coverage.observed_outside_scope_questions
        ),
        "outside_scope_objectives": len(
            report.coverage.observed_outside_scope_objectives
        ),
        "unlabeled_scope_excursions": unlabeled_scope_excursions,
        "gaps": [
            {
                "step": gap.step_index,
                "phase": gap.phase.value,
                "category": gap.category,
                "focus_objective_id": gap.focus_objective_id,
                "message": gap.message,
                "candidate_inventory": None,
                "candidate_inventory_note": (
                    "No decision was committed for this failed selection; "
                    "candidate count is unknown, not zero."
                ),
            }
            for gap in report.gaps
        ],
        "steps": steps,
    }


def _aggregate(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(run["topic_id"], run["profile"])].append(run)
    result: list[dict[str, Any]] = []
    for (topic_id, profile), rows in sorted(grouped.items()):
        all_steps = [step for row in rows for step in row["steps"]]
        gap_categories = Counter(
            gap["category"]
            for row in rows
            for gap in row["gaps"]
        )
        focused_steps = [
            step
            for step in all_steps
            if step["pedagogical_role"] != "exploration_probe"
        ]
        first_difficulties = [
            row["difficulty"]["first"]
            for row in rows
            if row["difficulty"]["first"] is not None
        ]
        candidate_counts = [
            int(
                step["candidate_inventory"][
                    "eligible_scored_candidate_count"
                ]
            )
            for step in all_steps
        ]
        complete_inventory_difficulties = [
            step["candidate_inventory"][
                "complete_inventory_difficulty"
            ]
            for step in all_steps
            if step["candidate_inventory"][
                "complete_inventory_difficulty"
            ]
            is not None
        ]
        focused_complete_inventory_difficulties = [
            step["candidate_inventory"][
                "complete_inventory_difficulty"
            ]
            for step in focused_steps
            if step["candidate_inventory"][
                "complete_inventory_difficulty"
            ]
            is not None
        ]
        result.append(
            {
                "topic_id": topic_id,
                "profile": profile,
                "trials": len(rows),
                "attempted": sum(row["attempted"] for row in rows),
                "correct": sum(row["correct"] for row in rows),
                "abstained": sum(row["abstained"] for row in rows),
                "mean_first_difficulty": (
                    statistics.fmean(first_difficulties)
                    if first_difficulties
                    else None
                ),
                "mean_served_difficulty": (
                    statistics.fmean(
                        float(step["difficulty"]) for step in all_steps
                    )
                    if all_steps
                    else None
                ),
                "mean_focused_difficulty": (
                    statistics.fmean(
                        float(step["difficulty"])
                        for step in focused_steps
                    )
                    if focused_steps
                    else None
                ),
                "mean_predicted_success": (
                    statistics.fmean(
                        float(step["predicted_success"]) for step in all_steps
                    )
                    if all_steps
                    else None
                ),
                "minimum_eligible_scored_candidates": (
                    min(candidate_counts) if candidate_counts else None
                ),
                "mean_eligible_scored_candidates": (
                    statistics.fmean(candidate_counts)
                    if candidate_counts
                    else None
                ),
                "complete_ranked_inventory_steps": sum(
                    bool(
                        step["candidate_inventory"][
                            "ranked_inventory_complete"
                        ]
                    )
                    for step in all_steps
                ),
                "complete_inventory_steps_without_introductory_candidate": sum(
                    difficulty["introductory_count"] == 0
                    for difficulty in complete_inventory_difficulties
                ),
                "focused_complete_inventory_steps_without_introductory_candidate": sum(
                    difficulty["introductory_count"] == 0
                    for difficulty in (
                        focused_complete_inventory_difficulties
                    )
                ),
                "mean_selected_difficulty_above_complete_inventory_minimum": (
                    statistics.fmean(
                        difficulty["selected_minus_minimum"]
                        for difficulty in complete_inventory_difficulties
                    )
                    if complete_inventory_difficulties
                    else None
                ),
                "below_full_sampling_frontier_steps": sum(
                    count < SAMPLING_FRONTIER_LIMIT
                    for count in candidate_counts
                ),
                "singleton_candidate_steps": sum(
                    count == 1 for count in candidate_counts
                ),
                "introductory_served": sum(
                    float(step["difficulty"])
                    < INTRODUCTORY_DIFFICULTY_CEILING
                    for step in all_steps
                ),
                "introductory_focused_served": sum(
                    float(step["difficulty"])
                    < INTRODUCTORY_DIFFICULTY_CEILING
                    for step in focused_steps
                ),
                "advanced_served": sum(
                    float(step["difficulty"]) > ADVANCED_DIFFICULTY_FLOOR
                    for step in all_steps
                ),
                "advanced_focused_served": sum(
                    float(step["difficulty"]) > ADVANCED_DIFFICULTY_FLOOR
                    for step in focused_steps
                ),
                "deliberate_exploration_questions": sum(
                    row["deliberate_exploration_questions"]
                    for row in rows
                ),
                "prerequisite_descents": sum(
                    row["prerequisite_descents"] for row in rows
                ),
                "gap_terminated_trials": sum(
                    bool(row["gaps"]) for row in rows
                ),
                "gap_categories": dict(sorted(gap_categories.items())),
            }
        )
    return result


def _run_once(
    *,
    corpus: Path,
    database_path: Path,
    topics: Sequence[str],
    seeds: Sequence[int],
    max_steps: int,
) -> dict[str, Any]:
    bundle = json.loads(corpus.read_text(encoding="utf-8"))
    objective_depth = _objective_depths(bundle)
    question_difficulty = {
        question["id"]: float(question["difficulty"])
        for question in bundle["questions"]
    }
    database = Database(database_path)
    database.initialize()
    database.import_corpus(
        *read_and_parse(corpus, include_catalog=True)
    )
    question_metadata = _release_question_metadata(database)
    engine = AdaptiveEngine(database)
    simulator = BehavioralSimulator(engine)
    runs: list[dict[str, Any]] = []
    run_index = 0
    for topic_id in topics:
        for profile in _profiles():
            for trial_index, seed in enumerate(seeds):
                learner_id = (
                    f"cold-{topic_id.removeprefix('t_')}-"
                    f"{profile.name}-{trial_index}"
                )
                report = simulator.run(
                    profile,
                    learner_id=learner_id,
                    root_concept_id=topic_id,
                    policy_seed=seed,
                    max_steps=max_steps,
                    mode="learn",
                    start_at=START + timedelta(days=run_index),
                    trial_index=trial_index,
                )
                runs.append(
                    _run_payload(
                        report=report,
                        database=database,
                        question_difficulty=question_difficulty,
                        question_metadata=question_metadata,
                        objective_depth=objective_depth,
                    )
                )
                run_index += 1

    integrity = database.verify_integrity()
    hard_invariants = {
        "integrity_ok": bool(integrity["ok"]),
        "no_unlabeled_scope_excursions": all(
            not any(run["unlabeled_scope_excursions"].values())
            for run in runs
        ),
        "no_remediation_exact_repeats": all(
            run["remediation_exact_repeats"] == 0 for run in runs
        ),
        "no_remediation_family_repeats": all(
            run["remediation_family_repeats"] == 0 for run in runs
        ),
    }
    return {
        "backend_versions": {
            "policy": POLICY_VERSION,
            "learner_model": engine.learner_model.model_version,
            "simulation_feedback_protocol": (
                SIMULATION_FEEDBACK_PROTOCOL_VERSION
            ),
            "semantic_projection_signature_schema": (
                SEMANTIC_PROJECTION_SIGNATURE_SCHEMA
            ),
        },
        "topics": [
            _corpus_topic_summary(bundle, topic_id) for topic_id in topics
        ],
        "objective_depths": dict(sorted(objective_depth.items())),
        "runs": runs,
        "aggregate": _aggregate(runs),
        "hard_invariants": hard_invariants,
        "integrity_errors": integrity["errors"],
    }


def run_cold_start_audit(
    *,
    corpus: Path = DEFAULT_CORPUS,
    topics: Sequence[str] = DEFAULT_TOPICS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    max_steps: int = DEFAULT_STEPS,
    replicate: bool = True,
) -> dict[str, Any]:
    if not topics or any(not isinstance(topic, str) or not topic for topic in topics):
        raise ValueError("At least one non-empty topic ID is required.")
    if not seeds or any(type(seed) is not int for seed in seeds):
        raise ValueError("At least one integer seed is required.")
    if type(max_steps) is not int or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer.")
    corpus = corpus.resolve()
    corpus_bytes = corpus.read_bytes()
    corpus_digest = hashlib.sha256(corpus_bytes).hexdigest()
    with tempfile.TemporaryDirectory(prefix="tsq-cold-start-") as directory:
        root = Path(directory)
        corpus_snapshot = root / "corpus-snapshot.json"
        corpus_snapshot.write_bytes(corpus_bytes)
        primary = _run_once(
            corpus=corpus_snapshot,
            database_path=root / "primary.db",
            topics=topics,
            seeds=seeds,
            max_steps=max_steps,
        )
        primary_signature = canonical_hash(primary)
        replication_signature = None
        deterministic = None
        if replicate:
            replication = _run_once(
                corpus=corpus_snapshot,
                database_path=root / "replica.db",
                topics=topics,
                seeds=seeds,
                max_steps=max_steps,
            )
            replication_signature = canonical_hash(replication)
            deterministic = replication == primary

    hard_invariants = dict(primary["hard_invariants"])
    if replicate:
        hard_invariants["deterministic_replication"] = bool(
            deterministic
        )
    artifact_core = {
        "lab_version": LAB_VERSION,
        "corpus_sha256": corpus_digest,
        "topics_requested": list(topics),
        "policy_seeds": list(seeds),
        "max_steps": max_steps,
        "profiles": [profile.name for profile in _profiles()],
        "candidate_inventory_contract": {
            "stage": (
                "Production-ranked scored candidates remaining after runtime "
                "eligibility and routing constraints, before randomized "
                f"top-{SAMPLING_FRONTIER_LIMIT} sampling."
            ),
            "raw_release_inventory": False,
            "stored_ranked_prefix_limit": CANDIDATE_PREFIX_LIMIT,
            "sampling_frontier_limit": SAMPLING_FRONTIER_LIMIT,
            "full_inventory_commitment": (
                "When the stored prefix is complete, the durable SHA-256 "
                "commits ordered question IDs, total scores rendered to 8 "
                "decimal places, raw exposure counts, diagnostic information "
                "rendered to 12 decimal places, and successful-family counts. "
                "It does not commit exact binary floats or every component "
                "score."
            ),
            "durable_candidate_digest_fields": [
                "question_id",
                "total",
                "coverage_raw_exposures",
                "coverage_diagnostic_information",
                "coverage_successful_retrieval_families",
            ],
            "durable_candidate_digest_encoding": {
                "total_decimal_places": 8,
                "coverage_diagnostic_information_decimal_places": 12,
                "integer_terms": "exact base-10 integers",
                "delimiter": "colon-separated fields; pipe-separated ranks",
            },
            "stored_prefix_artifact_hash": (
                "The lab separately hashes every stored prefix field for "
                "artifact replication; this is not a prior database "
                "commitment."
            ),
            "filter_attribution_boundary": (
                "This trace cannot identify which individual eligibility "
                "constraint removed a raw corpus item."
            ),
        },
        "replication_checked": replicate,
        "deterministic_replication": deterministic,
        "measurement_boundary": (
            "Synthetic cold-start behavior through the production engine. "
            "Authored difficulty and model-predicted success are uncalibrated "
            "diagnostics, not human ability or efficacy measurements."
        ),
        **primary,
        "hard_invariants": hard_invariants,
        "primary_signature": primary_signature,
        "replication_signature": replication_signature,
    }
    artifact = {
        **artifact_core,
        "status": (
            "ok" if all(hard_invariants.values()) else "invariant_failure"
        ),
    }
    artifact["artifact_sha256"] = canonical_hash(artifact)
    return artifact


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure TSQ cold-start difficulty on disposable databases."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--topic", action="append", dest="topics")
    parser.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stdout", action="store_true")
    parser.add_argument("--no-replication", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive.")
    artifact = run_cold_start_audit(
        corpus=args.corpus,
        topics=tuple(args.topics or DEFAULT_TOPICS),
        seeds=tuple(range(args.seeds)),
        max_steps=args.steps,
        replicate=not args.no_replication,
    )
    rendered = json.dumps(
        artifact,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    if args.stdout:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": artifact["status"],
                    "artifact_sha256": artifact["artifact_sha256"],
                    "output": str(args.output),
                    "runs": len(artifact["runs"]),
                },
                sort_keys=True,
            )
        )
    return 0 if artifact["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
