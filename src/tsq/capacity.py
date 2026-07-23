# SPDX-License-Identifier: MPL-2.0

"""Exact static capacity analysis for independent repair and verification.

The analyzer models a credible-correct stream of main questions.  Presenting a
main family consumes that family, while every possible wrong response must
still have a distinct repair family and verification family available.  It
reports the earliest and latest possible exhaustion over safe family-selection
orders; it does not claim that reserved families survive an actual remediation
episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from .graph import KnowledgeGraph
from .models import Concept, Misconception, Question, QuestionKind, Topic


ALGORITHM_VERSION = "sustained-serviceability-v2"
DEFAULT_STATE_LIMIT = 2_000_000
VERIFICATION_KINDS = frozenset(
    {
        QuestionKind.APPLICATION,
        QuestionKind.CALCULATION,
        QuestionKind.COMPARISON,
        QuestionKind.COUNTERFACTUAL,
        QuestionKind.DEBUGGING,
        QuestionKind.TRANSFER,
    }
)


class CapacityAnalysisLimitError(RuntimeError):
    """Raised rather than returning a heuristic when exact analysis is bounded."""

    def __init__(
        self, *, target_id: str, component_size: int, state_limit: int
    ) -> None:
        self.target_id = target_id
        self.component_size = component_size
        self.state_limit = state_limit
        super().__init__(
            f"Exact capacity analysis for {target_id} exceeded {state_limit} "
            f"states in a {component_size}-family dependency component."
        )


@dataclass(frozen=True, slots=True)
class CapacityTarget:
    target_id: str
    target_type: str
    owned_concept_ids: tuple[str, ...]
    target_main_count: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target_id, str) or not self.target_id.strip():
            raise ValueError("Capacity target_id must be a non-blank string.")
        if self.target_type not in {"concept", "topic"}:
            raise ValueError("Capacity target_type must be 'concept' or 'topic'.")
        if not self.owned_concept_ids or any(
            not isinstance(value, str) or not value.strip()
            for value in self.owned_concept_ids
        ):
            raise ValueError("Capacity targets require owned concept IDs.")
        canonical = tuple(sorted(set(self.owned_concept_ids)))
        object.__setattr__(self, "owned_concept_ids", canonical)
        if self.target_main_count is not None and (
            not isinstance(self.target_main_count, int)
            or isinstance(self.target_main_count, bool)
            or self.target_main_count < 1
        ):
            raise ValueError("target_main_count must be a positive integer or null.")

    def to_dict(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "target_type": self.target_type,
            "owned_concept_ids": list(self.owned_concept_ids),
            "target_main_count": self.target_main_count,
        }


@dataclass(frozen=True, slots=True)
class CapacityBlocker:
    path_kind: str
    question_id: str
    family_id: str
    misconception_id: str | None
    owner_concept_id: str
    objective_id: str | None
    remaining_repair_families: tuple[str, ...]
    remaining_verification_families: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path_kind": self.path_kind,
            "question_id": self.question_id,
            "family_id": self.family_id,
            "misconception_id": self.misconception_id,
            "owner_concept_id": self.owner_concept_id,
            "objective_id": self.objective_id,
            "remaining_repair_families": list(
                self.remaining_repair_families
            ),
            "remaining_verification_families": list(
                self.remaining_verification_families
            ),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CapacityWitness:
    consumed_family_ids: tuple[str, ...]
    terminal_main_family_ids: tuple[str, ...]
    blockers: tuple[CapacityBlocker, ...]

    @property
    def capacity(self) -> int:
        return len(self.consumed_family_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "consumed_family_ids": list(self.consumed_family_ids),
            "terminal_main_family_ids": list(self.terminal_main_family_ids),
            "blockers": [blocker.to_dict() for blocker in self.blockers],
        }


@dataclass(frozen=True, slots=True)
class OwnedConceptCapacity:
    concept_id: str
    eligible_question_count: int
    eligible_family_count: int
    initial_safe_family_count: int
    order_robust_main_capacity: int
    achievable_main_capacity: int
    target_main_count: int
    status: str
    states_evaluated: int

    def to_dict(self) -> dict[str, object]:
        return {
            "concept_id": self.concept_id,
            "eligible_questions": self.eligible_question_count,
            "eligible_families": self.eligible_family_count,
            "initial_safe_families": self.initial_safe_family_count,
            "order_robust_main_capacity": self.order_robust_main_capacity,
            "achievable_main_capacity": self.achievable_main_capacity,
            "target_main_count": self.target_main_count,
            "status": self.status,
            "states_evaluated": self.states_evaluated,
        }


@dataclass(frozen=True, slots=True)
class TargetCapacity:
    target: CapacityTarget
    scope_concept_ids: tuple[str, ...]
    eligible_question_count: int
    eligible_family_count: int
    available_scope_family_count: int
    initial_safe_family_ids: tuple[str, ...]
    order_robust_main_capacity: int
    achievable_main_capacity: int
    target_main_count: int
    recommended_with_headroom: int
    aggregate_status: str
    status: str
    states_evaluated: int
    component_count: int
    largest_component: int
    owned_concepts: tuple[OwnedConceptCapacity, ...]
    earliest_exhaustion: CapacityWitness
    maximum_capacity: CapacityWitness

    @property
    def order_loss(self) -> int:
        return self.achievable_main_capacity - self.order_robust_main_capacity

    @property
    def owned_concept_order_robust_floor(self) -> int:
        return min(
            (value.order_robust_main_capacity for value in self.owned_concepts),
            default=0,
        )

    @property
    def owned_concept_achievable_floor(self) -> int:
        return min(
            (value.achievable_main_capacity for value in self.owned_concepts),
            default=0,
        )

    @property
    def missing_owned_concept_ids(self) -> tuple[str, ...]:
        return tuple(
            value.concept_id
            for value in self.owned_concepts
            if value.eligible_family_count == 0
        )

    @property
    def thin_owned_concept_ids(self) -> tuple[str, ...]:
        return tuple(
            value.concept_id
            for value in self.owned_concepts
            if value.eligible_family_count > 0
            and value.status in {"blocked", "thin"}
        )

    @property
    def order_sensitive_owned_concept_ids(self) -> tuple[str, ...]:
        return tuple(
            value.concept_id
            for value in self.owned_concepts
            if value.status == "order_sensitive"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **self.target.to_dict(),
            "scope_concept_ids": list(self.scope_concept_ids),
            "eligible_questions": self.eligible_question_count,
            "eligible_families": self.eligible_family_count,
            "available_scope_families": self.available_scope_family_count,
            "initial_safe_families": list(self.initial_safe_family_ids),
            "order_robust_main_capacity": self.order_robust_main_capacity,
            "achievable_main_capacity": self.achievable_main_capacity,
            "order_loss": self.order_loss,
            "target_main_count": self.target_main_count,
            "recommended_with_headroom": self.recommended_with_headroom,
            "aggregate_status": self.aggregate_status,
            "status": self.status,
            "exact": True,
            "states_evaluated": self.states_evaluated,
            "component_count": self.component_count,
            "largest_component": self.largest_component,
            "owned_concept_order_robust_floor": (
                self.owned_concept_order_robust_floor
            ),
            "owned_concept_achievable_floor": self.owned_concept_achievable_floor,
            "missing_owned_concept_ids": list(self.missing_owned_concept_ids),
            "thin_owned_concept_ids": list(self.thin_owned_concept_ids),
            "order_sensitive_owned_concept_ids": list(
                self.order_sensitive_owned_concept_ids
            ),
            "owned_concepts": [
                concept.to_dict() for concept in self.owned_concepts
            ],
            "earliest_exhaustion": self.earliest_exhaustion.to_dict(),
            "maximum_capacity": self.maximum_capacity.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CapacityReport:
    targets: tuple[TargetCapacity, ...]
    algorithm: str = ALGORITHM_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "assumptions": {
                "family_reuse": "none_within_session",
                "depletion_model": "credible_correct_main_families_only",
                "repair_and_verification_must_be_distinct": True,
                "objective_paths": (
                    "direct objective and exact diagnostic objective/misconception"
                ),
                "legacy_paths": "primary concept and misconception owner",
                "verification_kinds": sorted(
                    value.value for value in VERIFICATION_KINDS
                ),
            },
            "targets": [target.to_dict() for target in self.targets],
        }


@dataclass(frozen=True, slots=True)
class _Requirement:
    path_kind: str
    misconception_id: str | None
    owner_concept_id: str
    objective_id: str | None
    repair_mask: int
    verification_mask: int


@dataclass(frozen=True, slots=True)
class _CompiledQuestion:
    question_id: str
    family_id: str
    family_bit: int
    requirements: tuple[_Requirement, ...]


@dataclass(frozen=True, slots=True)
class _Bounds:
    low: int
    high: int
    low_sequence: tuple[str, ...]
    high_sequence: tuple[str, ...]


class _StateBudget:
    def __init__(self, target_id: str, state_limit: int) -> None:
        self.target_id = target_id
        self.state_limit = state_limit
        self.used = 0

    def consume(self, component_size: int) -> None:
        self.used += 1
        if self.used > self.state_limit:
            raise CapacityAnalysisLimitError(
                target_id=self.target_id,
                component_size=component_size,
                state_limit=self.state_limit,
            )


def concept_target(
    concept_id: str, *, target_main_count: int | None = None
) -> CapacityTarget:
    return CapacityTarget(
        target_id=concept_id,
        target_type="concept",
        owned_concept_ids=(concept_id,),
        target_main_count=target_main_count,
    )


def topic_target(
    topic_id: str,
    topics: Iterable[Topic],
    *,
    target_main_count: int | None = None,
) -> CapacityTarget:
    topic_rows = tuple(topics)
    by_id = {topic.id: topic for topic in topic_rows}
    if topic_id not in by_id:
        raise ValueError(f"Unknown capacity topic: {topic_id}")
    selected = {topic_id}
    changed = True
    while changed:
        before = len(selected)
        selected.update(
            topic.id for topic in topic_rows if topic.parent_id in selected
        )
        changed = len(selected) != before
    owned = {
        concept_id
        for selected_id in selected
        for concept_id in by_id[selected_id].concept_ids
    }
    return CapacityTarget(
        target_id=topic_id,
        target_type="topic",
        owned_concept_ids=tuple(sorted(owned)),
        target_main_count=target_main_count,
    )


def catalog_targets(
    concepts: Iterable[Concept],
    topics: Iterable[Topic],
    *,
    include_concepts: bool = True,
    include_topics: bool = True,
) -> tuple[CapacityTarget, ...]:
    topic_rows = tuple(topics)
    results: list[CapacityTarget] = []
    if include_concepts:
        results.extend(concept_target(concept.id) for concept in concepts)
    if include_topics:
        results.extend(topic_target(topic.id, topic_rows) for topic in topic_rows)
    return tuple(sorted(results, key=lambda target: (target.target_type, target.target_id)))


def analyze_sustained_capacity(
    questions: Iterable[Question],
    graph: KnowledgeGraph,
    misconceptions: Iterable[Misconception],
    targets: Iterable[CapacityTarget],
    *,
    unavailable_family_ids: Iterable[str] = (),
    state_limit: int = DEFAULT_STATE_LIMIT,
) -> CapacityReport:
    """Analyze exact sequential main-family capacity for one or more targets."""

    if (
        not isinstance(state_limit, int)
        or isinstance(state_limit, bool)
        or state_limit < 1
    ):
        raise ValueError("state_limit must be a positive integer.")
    unavailable = frozenset(unavailable_family_ids)
    if any(not isinstance(value, str) or not value for value in unavailable):
        raise ValueError("unavailable_family_ids must contain non-blank strings.")
    active_questions = tuple(
        question
        for question in questions
        if question.status.eligible_for_adaptation
        and question.family_id not in unavailable
    )
    owner_by_misconception: dict[str, str] = {}
    for misconception in misconceptions:
        prior = owner_by_misconception.setdefault(
            misconception.id, misconception.concept_id
        )
        if prior != misconception.concept_id:
            raise ValueError(
                f"Misconception {misconception.id} has conflicting owners."
            )
    target_rows = tuple(targets)
    if len({target.target_id for target in target_rows}) != len(target_rows):
        raise ValueError("Capacity target IDs must be unique within one report.")
    concept_cache: dict[tuple[str, tuple[str, ...]], TargetCapacity] = {}
    results: list[TargetCapacity] = []
    for target in sorted(
        target_rows, key=lambda value: (value.target_type, value.target_id)
    ):
        scope = _target_scope(graph, target)
        child_results: tuple[TargetCapacity, ...] = ()
        if target.target_type == "topic":
            scope_key = tuple(sorted(scope))
            children: list[TargetCapacity] = []
            for concept_id in target.owned_concept_ids:
                key = (concept_id, scope_key)
                child = concept_cache.get(key)
                if child is None:
                    child = _analyze_target(
                        active_questions,
                        graph,
                        owner_by_misconception,
                        concept_target(concept_id),
                        state_limit,
                        scope_override=scope,
                    )
                    concept_cache[key] = child
                children.append(child)
            child_results = tuple(children)
        results.append(
            _analyze_target(
                active_questions,
                graph,
                owner_by_misconception,
                target,
                state_limit,
                scope_override=scope,
                owned_concept_results=child_results,
            )
        )
    return CapacityReport(tuple(results))


def _target_scope(graph: KnowledgeGraph, target: CapacityTarget) -> set[str]:
    owned = set(target.owned_concept_ids)
    unknown = owned - set(graph.concepts)
    if unknown:
        raise ValueError(
            f"Capacity target {target.target_id} has unknown concepts: "
            + ", ".join(sorted(unknown))
        )
    scope: set[str] = set()
    for concept_id in owned:
        scope.update(graph.learning_scope(concept_id))
    return scope


def _analyze_target(
    questions: tuple[Question, ...],
    graph: KnowledgeGraph,
    owner_by_misconception: dict[str, str],
    target: CapacityTarget,
    state_limit: int,
    *,
    scope_override: set[str] | None = None,
    owned_concept_results: tuple[TargetCapacity, ...] = (),
) -> TargetCapacity:
    owned = set(target.owned_concept_ids)
    scope = (
        set(scope_override)
        if scope_override is not None
        else _target_scope(graph, target)
    )
    if not owned <= scope:
        raise ValueError("Capacity support scope must contain every owned concept.")
    pool = tuple(
        question for question in questions if question.primary_concept_id in scope
    )
    pool_ids = {question.id for question in pool}
    eligible_questions = tuple(
        sorted(
            (
                question
                for question in pool
                if question.primary_concept_id in owned
            ),
            key=lambda question: (question.family_id, question.id),
        )
    )
    required_objective_ids = {
        objective_id
        for question in eligible_questions
        for objective_id in (
            question.objective_id,
            *(
                option.diagnostic_objective_id
                for option in question.options
            ),
        )
        if objective_id is not None
    }
    # Fine-objective repair is release-wide: a valid companion can use one of
    # the objective's declared supporting concepts as its primary retrieval
    # mapping and therefore sit outside the broad graph scope of the main item.
    reserve_questions = tuple(
        question
        for question in questions
        if question.objective_id in required_objective_ids
    )
    capacity_pool = tuple(
        {question.id: question for question in (*pool, *reserve_questions)}.values()
    )
    family_ids = tuple(
        sorted({question.family_id for question in capacity_pool})
    )
    family_index = {family_id: index for index, family_id in enumerate(family_ids)}

    primary_families: dict[str, set[str]] = {}
    misconception_families: dict[str, set[str]] = {}
    verification_families: dict[str, set[str]] = {}
    objective_families: dict[str, set[str]] = {}
    objective_verification_families: dict[str, set[str]] = {}
    objective_misconception_families: dict[
        tuple[str, str], set[str]
    ] = {}
    objective_misconception_verifications: dict[
        tuple[str, str], set[str]
    ] = {}
    objective_by_id = {
        question.objective.id: question.objective
        for question in questions
        if question.objective is not None
    }
    for question in capacity_pool:
        if question.id in pool_ids:
            primary_families.setdefault(
                question.primary_concept_id, set()
            ).add(question.family_id)
            if question.kind in VERIFICATION_KINDS:
                verification_families.setdefault(
                    question.primary_concept_id, set()
                ).add(question.family_id)
        if question.objective_id is not None:
            objective_families.setdefault(
                question.objective_id, set()
            ).add(question.family_id)
            if question.kind in VERIFICATION_KINDS:
                objective_verification_families.setdefault(
                    question.objective_id, set()
                ).add(question.family_id)
        for option in question.options:
            misconception_id = option.misconception_id
            if misconception_id is None:
                continue
            if misconception_id not in owner_by_misconception:
                raise ValueError(
                    f"Question {question.id} references unknown misconception "
                    f"{misconception_id}."
                )
            if question.id in pool_ids:
                misconception_families.setdefault(
                    misconception_id, set()
                ).add(question.family_id)
            diagnostic_objective_id = option.diagnostic_objective_id
            if (
                question.objective_id is not None
                and diagnostic_objective_id == question.objective_id
            ):
                pair = (question.objective_id, misconception_id)
                objective_misconception_families.setdefault(
                    pair, set()
                ).add(question.family_id)
                if question.kind in VERIFICATION_KINDS:
                    objective_misconception_verifications.setdefault(
                        pair, set()
                    ).add(question.family_id)

    def mask(values: Iterable[str]) -> int:
        result = 0
        for value in values:
            if value in family_index:
                result |= 1 << family_index[value]
        return result

    compiled_by_family: dict[str, list[_CompiledQuestion]] = {}
    for question in eligible_questions:
        primary = question.primary_concept_id
        if question.objective_id is not None:
            objective = objective_by_id[question.objective_id]
            requirements = [
                _Requirement(
                    "objective_generic",
                    None,
                    objective.primary_concept_id,
                    question.objective_id,
                    mask(
                        objective_families.get(
                            question.objective_id, ()
                        )
                    ),
                    mask(
                        objective_verification_families.get(
                            question.objective_id, ()
                        )
                    ),
                )
            ]
        else:
            requirements = [
                _Requirement(
                    "generic",
                    None,
                    primary,
                    None,
                    mask(primary_families.get(primary, ())),
                    mask(verification_families.get(primary, ())),
                )
            ]
        diagnostic_pairs = {
            (
                option.misconception_id,
                option.diagnostic_objective_id or question.objective_id,
            )
            for option in question.options
            if option.misconception_id is not None
        }
        for misconception_id, diagnostic_objective_id in sorted(
            diagnostic_pairs,
            key=lambda pair: (pair[0], pair[1] or ""),
        ):
            owner = owner_by_misconception[misconception_id]
            if diagnostic_objective_id is not None:
                if diagnostic_objective_id not in objective_by_id:
                    raise ValueError(
                        f"Question {question.id} references unknown diagnostic "
                        f"objective {diagnostic_objective_id}."
                    )
                pair = (diagnostic_objective_id, misconception_id)
                requirements.append(
                    _Requirement(
                        "objective_misconception",
                        misconception_id,
                        owner,
                        diagnostic_objective_id,
                        mask(
                            objective_misconception_families.get(
                                pair, ()
                            )
                        ),
                        mask(
                            objective_misconception_verifications.get(
                                pair, ()
                            )
                        ),
                    )
                )
            else:
                requirements.append(
                    _Requirement(
                        "misconception",
                        misconception_id,
                        owner,
                        None,
                        mask(
                            misconception_families.get(
                                misconception_id, ()
                            )
                        ),
                        mask(verification_families.get(owner, ())),
                    )
                )
        compiled_by_family.setdefault(question.family_id, []).append(
            _CompiledQuestion(
                question.id,
                question.family_id,
                1 << family_index[question.family_id],
                tuple(requirements),
            )
        )
    main_families = tuple(sorted(compiled_by_family))
    main_mask = mask(main_families)
    all_mask = (1 << len(family_ids)) - 1

    def question_safe(question: _CompiledQuestion, remaining_mask: int) -> bool:
        after = remaining_mask & ~question.family_bit
        for requirement in question.requirements:
            repairs = requirement.repair_mask & after
            verifications = requirement.verification_mask & after
            if (
                not repairs
                or not verifications
                or (repairs | verifications).bit_count() < 2
            ):
                return False
        return True

    initial_safe = tuple(
        family_id
        for family_id in main_families
        if any(
            question_safe(question, all_mask)
            for question in compiled_by_family[family_id]
        )
    )
    components = _dependency_components(
        main_families, compiled_by_family, main_mask, family_index
    )
    budget = _StateBudget(target.target_id, state_limit)
    low_sequences: list[tuple[str, ...]] = []
    high_sequences: list[tuple[str, ...]] = []
    low_total = 0
    high_total = 0
    for component in components:
        bounds = _component_bounds(
            component,
            all_mask=all_mask,
            family_index=family_index,
            compiled_by_family=compiled_by_family,
            question_safe=question_safe,
            budget=budget,
        )
        low_total += bounds.low
        high_total += bounds.high
        low_sequences.append(bounds.low_sequence)
        high_sequences.append(bounds.high_sequence)
    low_sequence = _merge_sequences(low_sequences)
    high_sequence = _merge_sequences(high_sequences)
    if len(low_sequence) != low_total or len(high_sequence) != high_total:
        raise AssertionError("Capacity witness length diverged from exact bounds.")

    low_witness = _witness(
        low_sequence,
        main_families,
        all_mask,
        family_index,
        compiled_by_family,
    )
    high_witness = _witness(
        high_sequence,
        main_families,
        all_mask,
        family_index,
        compiled_by_family,
    )
    assessed_owned = {
        question.primary_concept_id for question in eligible_questions
    }
    target_count = target.target_main_count
    if target_count is None:
        target_count = (
            3
            if target.target_type == "concept"
            else min(10, 3 * max(1, len(assessed_owned)))
        )
    recommended = target_count + 2
    aggregate_status = _capacity_status(
        low_total, high_total, target_count, recommended
    )
    if owned_concept_results:
        owned_summaries = tuple(
            _owned_concept_summary(result)
            for result in sorted(
                owned_concept_results,
                key=lambda value: value.target.target_id,
            )
        )
    elif target.target_type == "concept":
        owned_summaries = (
            OwnedConceptCapacity(
                concept_id=target.owned_concept_ids[0],
                eligible_question_count=len(eligible_questions),
                eligible_family_count=len(main_families),
                initial_safe_family_count=len(initial_safe),
                order_robust_main_capacity=low_total,
                achievable_main_capacity=high_total,
                target_main_count=target_count,
                status=aggregate_status,
                states_evaluated=budget.used,
            ),
        )
    else:
        raise AssertionError("Topic capacity requires owned-concept analyses.")
    status = _conservative_status(aggregate_status, owned_summaries)
    return TargetCapacity(
        target=target,
        scope_concept_ids=tuple(sorted(scope)),
        eligible_question_count=len(eligible_questions),
        eligible_family_count=len(main_families),
        available_scope_family_count=len(family_ids),
        initial_safe_family_ids=initial_safe,
        order_robust_main_capacity=low_total,
        achievable_main_capacity=high_total,
        target_main_count=target_count,
        recommended_with_headroom=recommended,
        aggregate_status=aggregate_status,
        status=status,
        states_evaluated=budget.used,
        component_count=len(components),
        largest_component=max((len(component) for component in components), default=0),
        owned_concepts=owned_summaries,
        earliest_exhaustion=low_witness,
        maximum_capacity=high_witness,
    )


def _capacity_status(
    low: int, high: int, target: int, recommended: int
) -> str:
    if high == 0:
        return "blocked"
    if low < target <= high:
        return "order_sensitive"
    if high < target:
        return "thin"
    if low < recommended:
        return "adequate"
    return "healthy"


def _owned_concept_summary(result: TargetCapacity) -> OwnedConceptCapacity:
    return OwnedConceptCapacity(
        concept_id=result.target.target_id,
        eligible_question_count=result.eligible_question_count,
        eligible_family_count=result.eligible_family_count,
        initial_safe_family_count=len(result.initial_safe_family_ids),
        order_robust_main_capacity=result.order_robust_main_capacity,
        achievable_main_capacity=result.achievable_main_capacity,
        target_main_count=result.target_main_count,
        status=result.status,
        states_evaluated=result.states_evaluated,
    )


def _conservative_status(
    aggregate_status: str,
    owned_concepts: tuple[OwnedConceptCapacity, ...],
) -> str:
    if any(value.eligible_family_count == 0 for value in owned_concepts):
        return "blocked"
    severity = {
        "healthy": 0,
        "adequate": 1,
        "order_sensitive": 2,
        "thin": 3,
        "blocked": 4,
    }
    return max(
        (aggregate_status, *(value.status for value in owned_concepts)),
        key=lambda status: severity[status],
    )


def _dependency_components(
    main_families: tuple[str, ...],
    compiled_by_family: dict[str, list[_CompiledQuestion]],
    main_mask: int,
    family_index: dict[str, int],
) -> tuple[tuple[str, ...], ...]:
    adjacency = {family_id: set() for family_id in main_families}
    family_by_index = {index: family_id for family_id, index in family_index.items()}
    for family_id in main_families:
        dependency_mask = 0
        for question in compiled_by_family[family_id]:
            for requirement in question.requirements:
                dependency_mask |= requirement.repair_mask
                dependency_mask |= requirement.verification_mask
        dependency_mask &= main_mask & ~(1 << family_index[family_id])
        while dependency_mask:
            bit = dependency_mask & -dependency_mask
            dependency = family_by_index[bit.bit_length() - 1]
            adjacency[family_id].add(dependency)
            adjacency[dependency].add(family_id)
            dependency_mask ^= bit
    seen: set[str] = set()
    components: list[tuple[str, ...]] = []
    for family_id in main_families:
        if family_id in seen:
            continue
        stack = [family_id]
        seen.add(family_id)
        component: list[str] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for dependency in sorted(adjacency[current], reverse=True):
                if dependency not in seen:
                    seen.add(dependency)
                    stack.append(dependency)
        components.append(tuple(sorted(component)))
    return tuple(sorted(components, key=lambda values: values[0]))


def _component_bounds(
    component: tuple[str, ...],
    *,
    all_mask: int,
    family_index: dict[str, int],
    compiled_by_family: dict[str, list[_CompiledQuestion]],
    question_safe,
    budget: _StateBudget,
) -> _Bounds:
    global_bits = tuple(1 << family_index[family_id] for family_id in component)
    dependency_masks = tuple(
        sorted(
            {
                mask
                for family_id in component
                for question in compiled_by_family[family_id]
                for requirement in question.requirements
                for mask in (
                    requirement.repair_mask,
                    requirement.verification_mask,
                )
            }
        )
    )

    # Families with the same selectable requirement templates and identical
    # membership in every reserve set are exact automorphisms of this state
    # machine.  Search only one deterministic representative for each such
    # group.  This quotients permutations of interchangeable families while
    # retaining concrete IDs in the reconstructed witness.
    equivalence: dict[tuple[object, ...], list[int]] = {}
    for index, family_id in enumerate(component):
        bit = global_bits[index]
        templates = tuple(
            sorted(
                {
                    tuple(
                        (
                            requirement.path_kind,
                            requirement.misconception_id or "",
                            requirement.owner_concept_id,
                            requirement.objective_id or "",
                            requirement.repair_mask,
                            requirement.verification_mask,
                        )
                        for requirement in question.requirements
                    )
                    for question in compiled_by_family[family_id]
                }
            )
        )
        signature: tuple[object, ...] = (
            tuple(bool(mask & bit) for mask in dependency_masks),
            templates,
        )
        equivalence.setdefault(signature, []).append(index)
    equivalent_groups = tuple(
        tuple(indices) for indices in equivalence.values()
    )
    local_group_masks = tuple(
        sum(1 << index for index in group)
        for group in equivalent_groups
    )
    local_group_prefixes = tuple(
        tuple(
            sum(1 << index for index in group[:count])
            for count in range(len(group) + 1)
        )
        for group in equivalent_groups
    )
    evaluated_states: set[int] = set()

    def charge_state(consumed: int) -> None:
        if consumed in evaluated_states:
            return
        budget.consume(len(component))
        evaluated_states.add(consumed)

    @lru_cache(maxsize=None)
    def safe_indices(consumed: int) -> tuple[int, ...]:
        removed = 0
        for index, bit in enumerate(global_bits):
            if consumed & (1 << index):
                removed |= bit
        remaining = all_mask & ~removed
        return tuple(
            index
            for index, family_id in enumerate(component)
            if not consumed & (1 << index)
            and any(
                question_safe(question, remaining)
                for question in compiled_by_family[family_id]
            )
        )

    def branch_indices(safe: tuple[int, ...]) -> tuple[int, ...]:
        safe_set = set(safe)
        return tuple(
            next(index for index in group if index in safe_set)
            for group in equivalent_groups
            if any(index in safe_set for index in group)
        )

    def canonicalize(consumed: int) -> int:
        """Return the representative state under proven family symmetries."""

        return sum(
            prefixes[(consumed & group_mask).bit_count()]
            for group_mask, prefixes in zip(
                local_group_masks, local_group_prefixes, strict=True
            )
        )

    def certified_equal_bound() -> tuple[int, tuple[str, ...]] | None:
        """Prove a common bound with mandatory-threshold certificates.

        Every repair/verification requirement induces a cardinality threshold
        on a reserve mask.  When every selectable family in that mask has the
        threshold in every variant, no valid sequence can consume more than
        ``mask_size - threshold`` of those families.  A disjoint cover of the
        component therefore supplies a rigorous packing upper bound.

        If a real greedy sequence reaches that bound, only the adversarial
        lower bound remains.  We prove it by searching for any smaller
        terminal set under the same necessary quotas.  This relaxation may
        consume families in an impossible order, so failure to find a set is
        a sound certificate that no real order can exhaust earlier.  Exact
        automorphisms canonicalize that small proof search.  If any part of
        the certificate fails, the exhaustive recurrence below remains the
        fail-closed fallback.
        """

        initial_safe = safe_indices(0)
        charge_state(0)
        if not initial_safe:
            return 0, ()
        universe_mask = sum(1 << index for index in initial_safe)
        universe_global_mask = sum(global_bits[index] for index in initial_safe)

        def localize(mask: int) -> int:
            return sum(
                1 << index
                for index, bit in enumerate(global_bits)
                if mask & bit
            )

        def question_thresholds(
            question: _CompiledQuestion,
        ) -> dict[int, int]:
            thresholds: dict[int, int] = {}
            for requirement in question.requirements:
                for reserve_mask, threshold in (
                    (requirement.repair_mask, 1),
                    (requirement.verification_mask, 1),
                    (
                        requirement.repair_mask
                        | requirement.verification_mask,
                        2,
                    ),
                ):
                    thresholds[reserve_mask] = max(
                        thresholds.get(reserve_mask, 0), threshold
                    )
            return thresholds

        thresholds_by_question = {
            question: question_thresholds(question)
            for family_id in component
            for question in compiled_by_family[family_id]
        }
        possible_thresholds = {
            (reserve_mask, threshold)
            for thresholds in thresholds_by_question.values()
            for reserve_mask, threshold in thresholds.items()
        }
        quota_by_block: dict[int, int] = {}
        for reserve_mask, threshold in possible_thresholds:
            fixed = (reserve_mask & ~universe_global_mask).bit_count()
            dynamic_threshold = threshold - fixed
            block = localize(reserve_mask) & universe_mask
            if dynamic_threshold <= 0 or not block:
                continue
            members = tuple(
                index
                for index in initial_safe
                if block & (1 << index)
            )
            mandatory = all(
                all(
                    thresholds_by_question[question].get(reserve_mask, 0)
                    >= threshold
                    for question in compiled_by_family[component[index]]
                )
                for index in members
            )
            if not mandatory:
                continue
            quota = len(members) - dynamic_threshold
            if quota < 0:
                continue
            quota_by_block[block] = min(
                quota_by_block.get(block, len(members)), quota
            )

        # Whole automorphism groups are always valid, symmetry-preserving
        # fallback blocks with the trivial one-use-per-family quota.
        for group_mask in local_group_masks:
            block = group_mask & universe_mask
            if block:
                quota_by_block.setdefault(block, block.bit_count())
        quota_candidates = tuple(
            sorted(
                quota_by_block.items(),
                key=lambda value: (
                    value[1],
                    -value[0].bit_count(),
                    value[0],
                ),
            )
        )
        candidates_by_index: dict[int, list[int]] = {
            index: [] for index in initial_safe
        }
        for candidate_index, (block, _) in enumerate(quota_candidates):
            for index in initial_safe:
                if block & (1 << index):
                    candidates_by_index[index].append(candidate_index)

        @lru_cache(maxsize=None)
        def best_partition(
            remaining: int,
        ) -> tuple[int, tuple[int, ...]] | None:
            if not remaining:
                return 0, ()
            first = (remaining & -remaining).bit_length() - 1
            best: tuple[int, tuple[int, ...]] | None = None
            for candidate_index in candidates_by_index[first]:
                block, quota = quota_candidates[candidate_index]
                if block & remaining != block:
                    continue
                child = best_partition(remaining ^ block)
                if child is None:
                    continue
                candidate = (
                    quota + child[0],
                    (candidate_index,) + child[1],
                )
                if best is None or candidate < best:
                    best = candidate
            return best

        partition = best_partition(universe_mask)
        if partition is None:
            return None
        partition_cost, partition_indices = partition
        quota_blocks = tuple(
            quota_candidates[index] for index in partition_indices
        )

        # Consume the lexicographically first safe family at each state.  If
        # its real terminal path reaches the packing bound, it witnesses the
        # exact achievable capacity and is the canonical sequence when the
        # lower-bound certificate also succeeds.
        consumed = 0
        greedy_sequence: list[str] = []
        while True:
            charge_state(consumed)
            safe = safe_indices(consumed)
            if not safe:
                break
            selected = min(safe, key=lambda index: component[index])
            greedy_sequence.append(component[selected])
            consumed = canonicalize(consumed | (1 << selected))
        if len(greedy_sequence) != partition_cost:
            return None

        def minimal_masks(values: Iterable[int]) -> tuple[int, ...]:
            kept: list[int] = []
            for value in sorted(
                set(values), key=lambda mask: (mask.bit_count(), mask)
            ):
                if any((prior & value) == prior for prior in kept):
                    continue
                kept.append(value)
            return tuple(kept)

        def requirement_failure_masks(
            question: _CompiledQuestion, requirement: _Requirement
        ) -> tuple[int, ...]:
            failures: list[int] = []
            question_bit = question.family_bit
            for reserve_mask in (
                requirement.repair_mask,
                requirement.verification_mask,
            ):
                without_question = reserve_mask & ~question_bit
                if not without_question & ~universe_global_mask:
                    failures.append(localize(without_question))
            union = (
                requirement.repair_mask | requirement.verification_mask
            ) & ~question_bit
            fixed_union = (union & ~universe_global_mask).bit_count()
            dynamic_union = localize(union) & universe_mask
            if fixed_union == 0:
                bits = tuple(
                    1 << index
                    for index in initial_safe
                    if dynamic_union & (1 << index)
                )
                if bits:
                    failures.extend(dynamic_union ^ bit for bit in bits)
                else:
                    failures.append(0)
            elif fixed_union == 1:
                failures.append(dynamic_union)
            return minimal_masks(failures)

        def family_disable_masks(index: int) -> tuple[int, ...]:
            combined: tuple[int, ...] = (0,)
            for question in compiled_by_family[component[index]]:
                question_failures = minimal_masks(
                    failure
                    for requirement in question.requirements
                    for failure in requirement_failure_masks(
                        question, requirement
                    )
                )
                if not question_failures:
                    combined = ()
                    break
                combined = minimal_masks(
                    prior | failure
                    for prior in combined
                    for failure in question_failures
                )
            return minimal_masks(((1 << index), *combined))

        disable_masks = {
            index: family_disable_masks(index) for index in initial_safe
        }

        def within_quotas(state: int) -> bool:
            return state.bit_count() < partition_cost and all(
                (state & block).bit_count() <= quota
                for block, quota in quota_blocks
            )

        @lru_cache(maxsize=None)
        def smaller_terminal_exists(state: int) -> bool:
            charge_state(state)
            safe = safe_indices(state)
            if not safe:
                return True
            choices: list[
                tuple[int, int, str, tuple[int, ...]]
            ] = []
            for index in branch_indices(safe):
                next_states = {
                    canonicalize(state | failure)
                    for failure in disable_masks[index]
                    if failure & ~state
                }
                next_states = {
                    next_state
                    for next_state in next_states
                    if next_state != state and within_quotas(next_state)
                }
                if not next_states:
                    return False
                choices.append(
                    (
                        len(next_states),
                        -min(
                            next_state.bit_count() - state.bit_count()
                            for next_state in next_states
                        ),
                        component[index],
                        tuple(
                            sorted(
                                next_states,
                                key=lambda next_state: (
                                    -(
                                        next_state.bit_count()
                                        - state.bit_count()
                                    ),
                                    next_state,
                                ),
                            )
                        ),
                    )
                )
            _, _, _, next_states = min(choices)
            return any(
                smaller_terminal_exists(next_state)
                for next_state in next_states
            )

        if smaller_terminal_exists(0):
            return None
        return partition_cost, tuple(greedy_sequence)

    certificate = certified_equal_bound()
    if certificate is not None:
        bound, sequence = certificate
        return _Bounds(bound, bound, sequence, sequence)

    @lru_cache(maxsize=None)
    def lowest(consumed: int) -> tuple[int, tuple[str, ...]]:
        charge_state(consumed)
        safe = safe_indices(consumed)
        if not safe:
            return 0, ()
        children = tuple(
            sorted(
                (
                    (
                        len(safe_indices(consumed | (1 << index))),
                        component[index],
                        index,
                    )
                    for index in branch_indices(safe)
                ),
                key=lambda value: (value[0], value[1]),
            )
        )
        # A removal that immediately disables every other family is an exact
        # one-step adversarial witness; avoid recursing merely to rediscover
        # the terminal state.
        if children[0][0] == 0:
            return 1, (children[0][1],)
        best: tuple[int, tuple[str, ...]] | None = None
        for _, family_id, index in children:
            child_count, child_sequence = lowest(
                consumed | (1 << index)
            )
            candidate = (
                1 + child_count,
                (family_id,) + child_sequence,
            )
            if best is None or candidate < best:
                best = candidate
            # Every non-terminal state must consume at least the selected
            # family. Once a one-step dead end is found, no branch can lower
            # the adversarial bound further.
            if best[0] == 1:
                break
        assert best is not None
        return best

    @lru_cache(maxsize=None)
    def highest(consumed: int) -> tuple[int, tuple[str, ...]]:
        charge_state(consumed)
        safe = safe_indices(consumed)
        if not safe:
            return 0, ()
        # Try the least destructive removal first.  When it preserves every
        # other currently-safe family, repeatedly doing so commonly proves
        # the monotone upper bound without exploring alternative orders.
        children = tuple(
            sorted(
                (
                    (
                        len(safe_indices(consumed | (1 << index))),
                        component[index],
                        index,
                    )
                    for index in branch_indices(safe)
                ),
                key=lambda value: (-value[0], value[1]),
            )
        )
        best: tuple[int, tuple[str, ...]] | None = None
        for _, family_id, index in children:
            child_count, child_sequence = highest(
                consumed | (1 << index)
            )
            candidate = (
                1 + child_count,
                (family_id,) + child_sequence,
            )
            if best is None or candidate[0] > best[0] or (
                candidate[0] == best[0]
                and candidate[1] < best[1]
            ):
                best = candidate
            # Safety is monotone under family removal: a family that is unsafe
            # now cannot become safe later. Consuming every currently safe
            # family therefore attains the exact upper bound for this state.
            if best[0] == len(safe):
                break
        assert best is not None
        return best

    low, low_sequence = lowest(0)
    high, high_sequence = highest(0)
    return _Bounds(low, high, low_sequence, high_sequence)


def _merge_sequences(sequences: Iterable[tuple[str, ...]]) -> tuple[str, ...]:
    pending = [list(sequence) for sequence in sequences if sequence]
    result: list[str] = []
    while pending:
        index = min(range(len(pending)), key=lambda value: pending[value][0])
        result.append(pending[index].pop(0))
        if not pending[index]:
            pending.pop(index)
    return tuple(result)


def _witness(
    sequence: tuple[str, ...],
    main_families: tuple[str, ...],
    all_mask: int,
    family_index: dict[str, int],
    compiled_by_family: dict[str, list[_CompiledQuestion]],
) -> CapacityWitness:
    removed = 0
    for family_id in sequence:
        removed |= 1 << family_index[family_id]
    remaining = all_mask & ~removed
    terminal = tuple(sorted(set(main_families) - set(sequence)))
    family_by_index = {index: family_id for family_id, index in family_index.items()}

    def names(mask: int) -> tuple[str, ...]:
        values: list[str] = []
        while mask:
            bit = mask & -mask
            values.append(family_by_index[bit.bit_length() - 1])
            mask ^= bit
        return tuple(sorted(values))

    blockers: list[CapacityBlocker] = []
    for family_id in terminal:
        for question in sorted(
            compiled_by_family[family_id], key=lambda value: value.question_id
        ):
            after = remaining & ~question.family_bit
            for requirement in question.requirements:
                repairs = requirement.repair_mask & after
                verifications = requirement.verification_mask & after
                if repairs and verifications and (repairs | verifications).bit_count() >= 2:
                    continue
                if not repairs:
                    reason = "no_repair_family"
                elif not verifications:
                    reason = "no_verification_family"
                else:
                    reason = "repair_verification_not_independent"
                blockers.append(
                    CapacityBlocker(
                        path_kind=requirement.path_kind,
                        question_id=question.question_id,
                        family_id=family_id,
                        misconception_id=requirement.misconception_id,
                        owner_concept_id=requirement.owner_concept_id,
                        objective_id=requirement.objective_id,
                        remaining_repair_families=names(repairs),
                        remaining_verification_families=names(verifications),
                        reason=reason,
                    )
                )
    blockers.sort(
        key=lambda value: (
            value.family_id,
            value.question_id,
            value.path_kind,
            value.objective_id or "",
            value.misconception_id or "",
        )
    )
    return CapacityWitness(sequence, terminal, tuple(blockers))
