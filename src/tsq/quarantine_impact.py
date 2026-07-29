# SPDX-License-Identifier: MPL-2.0

"""Exact, non-activating capacity impact for quarantined questions.

This module answers a narrow authoring question: if specific quarantined
questions whose primary concept is owned by the selected target eventually
survive independent human review, which smallest review sets could close a
configured sustained-capacity gap? Candidate sets contain at most one question
from any semantic family. Prerequisite-scope and release-wide objective-reserve
candidates are intentionally outside this report.

The calculation replaces ``Question.status`` only on frozen in-memory copies
and then invokes the production exact capacity analyzer. It never serializes
the replacement, mutates a corpus, records learner evidence, establishes
factual correctness, or establishes semantic family independence.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from typing import Iterable, Sequence

from .capacity import (
    ALGORITHM_VERSION as CAPACITY_ALGORITHM_VERSION,
    DEFAULT_STATE_LIMIT,
    CapacityTarget,
    TargetCapacity,
    analyze_sustained_capacity,
)
from .evidence import canonical_digest
from .graph import KnowledgeGraph
from .models import (
    LearningObjective,
    Misconception,
    Question,
    QuestionStatus,
)
from .provenance import legacy_question_identity_payload


REPORT_SCHEMA = "quarantine-capacity-impact-v1"
MAX_COMBINATION_SIZE = 3
MAX_CANDIDATE_LIMIT = 32
MAX_RESULT_LIMIT = 100
MAX_EVALUATION_LIMIT = 2_000

_CLOSING_STATUSES = frozenset({"adequate", "healthy"})
_STATUS_RANK = {
    "blocked": 0,
    "thin": 1,
    "order_sensitive": 2,
    "adequate": 3,
    "healthy": 4,
}


def _objective_payload(
    objective: LearningObjective | None,
) -> dict[str, object] | None:
    if objective is None:
        return None
    return {
        "id": objective.id,
        "name": objective.name,
        "description": objective.description,
        "primary_concept_id": objective.primary_concept_id,
        "supporting_concept_ids": list(
            objective.supporting_concept_ids
        ),
        "operation": objective.operation.value,
        "evidence_type": objective.evidence_type,
        "prior_mastery": float(objective.prior_mastery),
        "objective_graph_version": objective.objective_graph_version,
        "prerequisites": [
            {
                "id": edge.id,
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "relation": edge.relation.value,
                "weight": float(edge.weight),
                "rationale": edge.rationale,
            }
            for edge in objective.prerequisites
        ],
    }


def _question_input_payload(question: Question) -> dict[str, object]:
    """Return every question term that can affect or contextualize the report."""

    identity = legacy_question_identity_payload(
        question_id=question.id,
        version=question.version,
        family_id=question.family_id,
        stem=question.stem,
        kind=question.kind.value,
        difficulty=question.difficulty,
        discrimination=question.discrimination,
        guess_rate=question.guess_rate,
        slip_rate=question.slip_rate,
        concepts=(
            (
                mapping.concept_id,
                mapping.weight,
                mapping.role.value,
            )
            for mapping in question.concepts
        ),
        options=(
            (
                option.id,
                option.text,
                option.correct,
                option.rationale,
                option.misconception_id,
                option.diagnostic_objective_id,
            )
            for option in question.options
        ),
        source_ids=question.source_ids,
        provenance=question.provenance,
        tags=question.tags,
        revision_of=question.revision_of,
        learning_objective_id=question.objective_id,
    )
    return {
        "identity": identity,
        "status": question.status.value,
        "eligible_for_adaptation": (
            question.status.eligible_for_adaptation
        ),
        "objective": _objective_payload(question.objective),
    }


def _graph_input_payload(graph: KnowledgeGraph) -> dict[str, object]:
    return {
        "concepts": [
            {
                "id": concept.id,
                "name": concept.name,
                "description": concept.description,
                "domain": concept.domain,
                "prior_mastery": float(concept.prior_mastery),
            }
            for concept in sorted(
                graph.concepts.values(), key=lambda value: value.id
            )
        ],
        "edges": [
            {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "relation": edge.relation.value,
                "weight": float(edge.weight),
            }
            for edge in sorted(
                graph.edges,
                key=lambda value: (
                    value.source_id,
                    value.target_id,
                    value.relation.value,
                    value.weight,
                ),
            )
        ],
    }


def _misconception_input_payload(
    misconceptions: Sequence[Misconception],
) -> list[dict[str, object]]:
    return [
        {
            "id": misconception.id,
            "concept_id": misconception.concept_id,
            "name": misconception.name,
            "description": misconception.description,
        }
        for misconception in sorted(
            misconceptions, key=lambda value: value.id
        )
    ]


def _admissible_subset_count(
    candidates: Sequence[Question],
    maximum_size: int,
) -> int:
    """Count question subsets with at most one member from each family."""

    family_sizes: dict[str, int] = {}
    for question in candidates:
        family_sizes[question.family_id] = (
            family_sizes.get(question.family_id, 0) + 1
        )
    counts = [0] * (maximum_size + 1)
    counts[0] = 1
    for family_size in family_sizes.values():
        for size in range(maximum_size, 0, -1):
            counts[size] += counts[size - 1] * family_size
    return sum(counts[1:])


def _capacity_snapshot(result: TargetCapacity) -> dict[str, object]:
    return {
        "status": result.status,
        "aggregate_status": result.aggregate_status,
        "target_main_count": result.target_main_count,
        "order_robust_main_capacity": result.order_robust_main_capacity,
        "achievable_main_capacity": result.achievable_main_capacity,
        "initial_safe_family_count": len(result.initial_safe_family_ids),
        "owned_concept_order_robust_floor": (
            result.owned_concept_order_robust_floor
        ),
        "owned_concept_achievable_floor": (
            result.owned_concept_achievable_floor
        ),
        "thin_owned_concept_ids": list(result.thin_owned_concept_ids),
        "order_sensitive_owned_concept_ids": list(
            result.order_sensitive_owned_concept_ids
        ),
        "missing_owned_concept_ids": list(
            result.missing_owned_concept_ids
        ),
        "owned_concepts": [
            {
                "concept_id": value.concept_id,
                "status": value.status,
                "eligible_families": value.eligible_family_count,
                "initial_safe_families": value.initial_safe_family_count,
                "order_robust_main_capacity": (
                    value.order_robust_main_capacity
                ),
                "achievable_main_capacity": (
                    value.achievable_main_capacity
                ),
                "target_main_count": value.target_main_count,
            }
            for value in result.owned_concepts
        ],
    }


def _impact(
    result: TargetCapacity,
    baseline: TargetCapacity,
) -> dict[str, object]:
    baseline_thin = set(baseline.thin_owned_concept_ids)
    current_thin = set(result.thin_owned_concept_ids)
    return {
        "capacity": _capacity_snapshot(result),
        "delta": {
            "order_robust_main_capacity": (
                result.order_robust_main_capacity
                - baseline.order_robust_main_capacity
            ),
            "achievable_main_capacity": (
                result.achievable_main_capacity
                - baseline.achievable_main_capacity
            ),
            "initial_safe_family_count": (
                len(result.initial_safe_family_ids)
                - len(baseline.initial_safe_family_ids)
            ),
            "owned_concept_order_robust_floor": (
                result.owned_concept_order_robust_floor
                - baseline.owned_concept_order_robust_floor
            ),
            "owned_concept_achievable_floor": (
                result.owned_concept_achievable_floor
                - baseline.owned_concept_achievable_floor
            ),
        },
        "resolved_thin_owned_concept_ids": sorted(
            baseline_thin - current_thin
        ),
        "closes_configured_target": result.status in _CLOSING_STATUSES,
    }


def _candidate_payload(question: Question) -> dict[str, object]:
    provenance = question.provenance
    return {
        "question_id": question.id,
        "question_version": question.version,
        "family_id": question.family_id,
        "primary_concept_id": question.primary_concept_id,
        "learning_objective_id": question.objective_id,
        "kind": question.kind.value,
        "difficulty": float(question.difficulty),
        "source_ids": list(question.source_ids),
        "question_input_digest": canonical_digest(
            _question_input_payload(question)
        ),
        "original_question_status": question.status.value,
        "original_question_eligible_for_adaptation": (
            question.status.eligible_for_adaptation
        ),
        "counterfactual_status": QuestionStatus.APPROVED.value,
        "provenance_claims": {
            "generated": provenance.get("generated"),
            "provider": provenance.get("provider"),
            "batch_id": provenance.get("batch_id"),
            "review_status": provenance.get("review_status"),
            "human_review": provenance.get("human_review"),
            "human_review_status": provenance.get(
                "human_review_status"
            ),
            "activation": provenance.get("activation"),
        },
    }


def _combination_sort_key(
    row: dict[str, object],
) -> tuple[object, ...]:
    capacity = row["impact"]["capacity"]  # type: ignore[index]
    delta = row["impact"]["delta"]  # type: ignore[index]
    return (
        -_STATUS_RANK[str(capacity["status"])],
        -int(capacity["order_robust_main_capacity"]),
        -int(capacity["owned_concept_order_robust_floor"]),
        -int(capacity["achievable_main_capacity"]),
        -int(delta["initial_safe_family_count"]),
        len(row["question_ids"]),  # type: ignore[arg-type]
        tuple(row["question_ids"]),  # type: ignore[arg-type]
    )


def analyze_quarantine_capacity_impact(
    questions: Iterable[Question],
    graph: KnowledgeGraph,
    misconceptions: Iterable[Misconception],
    target: CapacityTarget,
    *,
    maximum_combination_size: int = 2,
    candidate_limit: int = 16,
    result_limit: int = 20,
    evaluation_limit: int = 500,
    state_limit: int = DEFAULT_STATE_LIMIT,
) -> dict[str, object]:
    """Rank exact structural capacity impact without activating content.

    Candidate subsets are exhausted in increasing cardinality. Once a closing
    cardinality is found, every subset of that size has been evaluated and
    larger subsets are intentionally omitted because the minimum is already
    exact.
    """

    if (
        not isinstance(maximum_combination_size, int)
        or isinstance(maximum_combination_size, bool)
        or not 1 <= maximum_combination_size <= MAX_COMBINATION_SIZE
    ):
        raise ValueError(
            "maximum_combination_size must be between 1 and "
            f"{MAX_COMBINATION_SIZE}."
        )
    if (
        not isinstance(candidate_limit, int)
        or isinstance(candidate_limit, bool)
        or not 1 <= candidate_limit <= MAX_CANDIDATE_LIMIT
    ):
        raise ValueError(
            f"candidate_limit must be between 1 and {MAX_CANDIDATE_LIMIT}."
        )
    if (
        not isinstance(result_limit, int)
        or isinstance(result_limit, bool)
        or not 1 <= result_limit <= MAX_RESULT_LIMIT
    ):
        raise ValueError(
            f"result_limit must be between 1 and {MAX_RESULT_LIMIT}."
        )
    if (
        not isinstance(evaluation_limit, int)
        or isinstance(evaluation_limit, bool)
        or not 1 <= evaluation_limit <= MAX_EVALUATION_LIMIT
    ):
        raise ValueError(
            "evaluation_limit must be between 1 and "
            f"{MAX_EVALUATION_LIMIT}."
        )
    if (
        not isinstance(state_limit, int)
        or isinstance(state_limit, bool)
        or not 1 <= state_limit <= DEFAULT_STATE_LIMIT
    ):
        raise ValueError(
            f"state_limit must be between 1 and {DEFAULT_STATE_LIMIT}."
        )
    if not isinstance(target, CapacityTarget):
        raise ValueError("target must be a CapacityTarget.")

    question_rows = tuple(questions)
    misconception_rows = tuple(misconceptions)
    question_ids = [question.id for question in question_rows]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError(
            "Quarantine impact requires globally unique question IDs."
        )
    misconception_ids = [
        misconception.id for misconception in misconception_rows
    ]
    if len(set(misconception_ids)) != len(misconception_ids):
        raise ValueError(
            "Quarantine impact requires globally unique misconception IDs."
        )
    objective_payloads: dict[str, dict[str, object]] = {}
    for question in question_rows:
        if question.objective is None:
            continue
        payload = _objective_payload(question.objective)
        if payload is None:
            raise AssertionError("Objective payload unexpectedly absent.")
        prior = objective_payloads.setdefault(
            question.objective.id, payload
        )
        if prior != payload:
            raise ValueError(
                "Quarantine impact found conflicting definitions for "
                f"learning objective {question.objective.id}."
            )
    analysis_input_manifest_digest = canonical_digest(
        {
            "schema": "quarantine-capacity-impact-input-v1",
            "capacity_algorithm_version": CAPACITY_ALGORITHM_VERSION,
            "target": target.to_dict(),
            "state_limit": state_limit,
            "maximum_combination_size": maximum_combination_size,
            "candidate_limit": candidate_limit,
            "result_limit": result_limit,
            "evaluation_limit": evaluation_limit,
            "questions": [
                _question_input_payload(question)
                for question in sorted(
                    question_rows,
                    key=lambda value: (
                        value.id,
                        value.version,
                        value.family_id,
                        value.status.value,
                    ),
                )
            ],
            "graph": _graph_input_payload(graph),
            "misconceptions": _misconception_input_payload(
                misconception_rows
            ),
        }
    )
    baseline = analyze_sustained_capacity(
        question_rows,
        graph,
        misconception_rows,
        (target,),
        state_limit=state_limit,
    ).targets[0]
    # Keep the authoring question narrow and auditable: rank only quarantined
    # items whose primary retrieval mapping belongs to the selected target.
    # Prerequisite-scope candidates can also affect reserve structure, but
    # mixing them into this report would turn a focused topic review into a
    # potentially large cross-curriculum search.
    owned = set(target.owned_concept_ids)
    candidates = tuple(
        sorted(
            (
                question
                for question in question_rows
                if question.status is QuestionStatus.QUARANTINED
                and question.primary_concept_id in owned
            ),
            key=lambda question: question.id,
        )
    )
    if len(candidates) > candidate_limit:
        raise ValueError(
            "Exact quarantine impact refuses to truncate "
            f"{len(candidates)} candidates at limit {candidate_limit}; "
            "increase --candidate-limit or narrow the target."
        )
    source_statuses = {
        question.id: question.status for question in candidates
    }
    distinct_candidate_family_count = len(
        {question.family_id for question in candidates}
    )
    preflight_depth = min(
        maximum_combination_size,
        len(candidates),
    )
    if baseline.status in _CLOSING_STATUSES and preflight_depth > 1:
        preflight_depth = 1
    preflight_evaluation_count = _admissible_subset_count(
        candidates,
        preflight_depth,
    )
    if preflight_evaluation_count > evaluation_limit:
        raise ValueError(
            "Exact quarantine impact requires "
            f"{preflight_evaluation_count} admissible candidate-subset "
            f"evaluations through size {preflight_depth}, exceeding "
            f"evaluation_limit {evaluation_limit}; narrow the target or "
            "combination bound. No heuristic or partial result was used."
        )

    def evaluate(subset: Sequence[Question]) -> TargetCapacity:
        promoted_ids = {question.id for question in subset}
        counterfactual = tuple(
            replace(question, status=QuestionStatus.APPROVED)
            if question.id in promoted_ids
            else question
            for question in question_rows
        )
        return analyze_sustained_capacity(
            counterfactual,
            graph,
            misconception_rows,
            (target,),
            state_limit=state_limit,
        ).targets[0]

    evaluated: list[dict[str, object]] = []
    repeated_family_subsets_skipped = 0
    baseline_closes = baseline.status in _CLOSING_STATUSES
    closing_at_minimum: list[dict[str, object]] = (
        [
            {
                "question_ids": [],
                "family_ids": [],
                "impact": _impact(baseline, baseline),
            }
        ]
        if baseline_closes
        else []
    )
    minimum_closing_size = 0 if baseline_closes else None
    largest_evaluated_size = 0
    largest_enumerated_size = 0
    for size in range(
        1, min(maximum_combination_size, len(candidates)) + 1
    ):
        largest_enumerated_size = size
        size_rows: list[dict[str, object]] = []
        for subset in combinations(candidates, size):
            family_ids = tuple(question.family_id for question in subset)
            if len(set(family_ids)) != len(family_ids):
                repeated_family_subsets_skipped += 1
                continue
            result = evaluate(subset)
            row = {
                "question_ids": [
                    question.id for question in subset
                ],
                "family_ids": list(family_ids),
                "impact": _impact(result, baseline),
            }
            size_rows.append(row)
        evaluated.extend(size_rows)
        if size_rows:
            largest_evaluated_size = size
        if baseline_closes:
            # Single-candidate impacts are still useful for review ordering;
            # larger sets cannot improve the already exact minimum of zero.
            break
        if minimum_closing_size is None:
            closing = [
                row
                for row in size_rows
                if row["impact"]["closes_configured_target"]  # type: ignore[index]
            ]
            if closing:
                minimum_closing_size = size
                closing_at_minimum = closing
                break

    if any(
        question.status is not source_statuses[question.id]
        for question in candidates
    ):
        raise AssertionError(
            "Quarantine impact mutated a source question status."
        )

    by_single_id = {
        str(row["question_ids"][0]): row
        for row in evaluated
        if len(row["question_ids"]) == 1  # type: ignore[arg-type]
    }
    candidate_rows = []
    for candidate in candidates:
        payload = _candidate_payload(candidate)
        payload["single_candidate_impact"] = by_single_id[candidate.id][
            "impact"
        ]
        candidate_rows.append(payload)

    ranked = sorted(evaluated, key=_combination_sort_key)
    closing_ranked = sorted(
        closing_at_minimum, key=_combination_sort_key
    )
    if baseline_closes:
        search_outcome = "live_target_already_met"
    elif minimum_closing_size is not None:
        search_outcome = "minimal_closing_combination_found"
    elif not candidates:
        search_outcome = "no_target_owned_quarantined_candidates"
    else:
        search_outcome = "no_closing_combination_within_bound"
    admissible_search_space_fully_exhausted = (
        largest_enumerated_size >= distinct_candidate_family_count
        or distinct_candidate_family_count == 0
    )
    closure_absence_exact = (
        minimum_closing_size is None
        and admissible_search_space_fully_exhausted
    )
    if closure_absence_exact and candidates:
        search_outcome = "no_closing_admissible_combination"
    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "algorithm": "exact-status-substitution-over-sustained-serviceability-v1",
        "capacity_algorithm_version": CAPACITY_ALGORITHM_VERSION,
        "analysis_input_manifest_digest": (
            analysis_input_manifest_digest
        ),
        "analysis_input_question_count": len(question_rows),
        "analysis_input_concept_count": len(graph.concepts),
        "analysis_input_edge_count": len(graph.edges),
        "analysis_input_misconception_count": len(
            misconception_rows
        ),
        "target": target.to_dict(),
        "baseline": _capacity_snapshot(baseline),
        "candidate_scope": "target-owned-primary-concepts-only",
        "candidate_combination_constraint": (
            "at-most-one-question-per-family"
        ),
        "candidate_count": len(candidates),
        "distinct_candidate_family_count": (
            distinct_candidate_family_count
        ),
        "candidate_limit": candidate_limit,
        "maximum_combination_size": maximum_combination_size,
        "state_limit_per_capacity_analysis": state_limit,
        "evaluation_limit": evaluation_limit,
        "preflight_combination_depth": preflight_depth,
        "preflight_admissible_combination_count": (
            preflight_evaluation_count
        ),
        "result_limit": result_limit,
        "largest_enumerated_combination_size": (
            largest_enumerated_size
        ),
        "largest_evaluated_combination_size": largest_evaluated_size,
        "evaluated_combination_count": len(evaluated),
        "repeated_family_subsets_skipped": (
            repeated_family_subsets_skipped
        ),
        "minimal_closing_combination_size": minimum_closing_size,
        "minimal_closing_size_is_exact_within_declared_search_space": (
            minimum_closing_size is not None
        ),
        "admissible_candidate_search_space_fully_exhausted": (
            admissible_search_space_fully_exhausted
        ),
        "closure_absence_is_exact_within_declared_candidate_space": (
            closure_absence_exact
        ),
        "search_outcome": search_outcome,
        "closing_combination_count_at_minimum": len(
            closing_at_minimum
        ),
        "closing_combinations": closing_ranked[:result_limit],
        "closing_combinations_truncated": (
            len(closing_ranked) > result_limit
        ),
        "best_available_combinations": ranked[:result_limit],
        "best_available_combinations_truncated": (
            len(ranked) > result_limit
        ),
        "candidates": candidate_rows,
        "candidate_set_digest": canonical_digest(
            [
                {
                    "question_id": question.id,
                    "family_id": question.family_id,
                    "status": question.status.value,
                    "primary_concept_id": question.primary_concept_id,
                    "question_input_digest": canonical_digest(
                        _question_input_payload(question)
                    ),
                }
                for question in candidates
            ]
        ),
        "boundary": {
            "counterfactual_only": True,
            "original_question_status_required": (
                QuestionStatus.QUARANTINED.value
            ),
            "source_corpus_mutated_by_analyzer": False,
            "database_opened_by_analyzer": False,
            "activation_performed_by_analyzer": False,
            "learner_evidence_recorded_by_analyzer": False,
            "human_review_required_before_activation": True,
            "factual_correctness_established": False,
            "source_grounding_established": False,
            "semantic_family_independence_established": False,
            "psychometric_validity_established": False,
            "capacity_results_exact_under_declared_structural_assumptions": True,
            "minimality_claim_limited_to": (
                "admissible target-owned quarantined question subsets with "
                "at most one question per family through the declared size "
                "bound"
            ),
            "prerequisite_scope_candidates_ranked": False,
            "release_wide_objective_reserve_candidates_ranked": False,
            "same_family_multi_question_subsets_ranked": False,
            "status_substitution": (
                "quarantined -> approved on frozen in-memory copies only"
            ),
        },
    }
    report["report_digest"] = canonical_digest(report)
    return report


__all__ = [
    "MAX_CANDIDATE_LIMIT",
    "MAX_COMBINATION_SIZE",
    "MAX_EVALUATION_LIMIT",
    "MAX_RESULT_LIMIT",
    "REPORT_SCHEMA",
    "analyze_quarantine_capacity_impact",
]
