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
from typing import Callable, Iterable

from .graph import KnowledgeGraph
from .models import Concept, Misconception, Question, QuestionKind, Topic


ALGORITHM_VERSION = "sustained-serviceability-v2"
DEFAULT_STATE_LIMIT = 2_000_000
_CLOSURE_SOLVER_MIN_COMPONENT_SIZE = 18
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


def _maximum_quota_packing(
    safe_mask: int,
    constraints: Iterable[tuple[int, int]],
    *,
    charge: Callable[[tuple[object, ...]], None] | None = None,
) -> int:
    """Return the exact 0/1 cardinality admitted by overlapping quotas.

    Variables with identical constraint incidence are interchangeable and can
    be grouped by multiplicity. Independent incidence components are solved
    separately. The residual-capacity dynamic program is exact for this quota
    relaxation; because real safe-removal paths satisfy every quota, its value
    is a rigorous upper bound on future consumptions.
    """

    quota_by_mask: dict[int, int] = {}
    for mask, quota in constraints:
        block = mask & safe_mask
        if not block:
            continue
        if quota < 0:
            return -1
        bounded_quota = min(quota, block.bit_count())
        quota_by_mask[block] = min(
            quota_by_mask.get(block, block.bit_count()),
            bounded_quota,
        )
    rows = tuple(
        sorted(
            (
                (mask, quota)
                for mask, quota in quota_by_mask.items()
                if quota < mask.bit_count()
            ),
            key=lambda value: (value[1], value[0].bit_count(), value[0]),
        )
    )
    if not rows:
        return safe_mask.bit_count()

    # A single quota A implies quota B when even filling every B-only variable
    # after saturating A cannot exceed B's allowance. Removing such rows keeps
    # the feasible 0/1 region unchanged and shrinks incidence vectors.
    def implies(
        source: tuple[int, int], target: tuple[int, int]
    ) -> bool:
        source_mask, source_quota = source
        target_mask, target_quota = target
        maximum_target = (
            (target_mask & ~source_mask).bit_count()
            + min(
                source_quota,
                (source_mask & target_mask).bit_count(),
            )
        )
        return maximum_target <= target_quota

    reduced_rows = tuple(
        row
        for index, row in enumerate(rows)
        if not any(
            other_index != index and implies(other, row)
            for other_index, other in enumerate(rows)
        )
    )
    if not reduced_rows:
        return safe_mask.bit_count()

    grouped: dict[tuple[int, ...], int] = {}
    free = 0
    pending = safe_mask
    while pending:
        bit = pending & -pending
        incidence = tuple(
            index
            for index, (mask, _) in enumerate(reduced_rows)
            if mask & bit
        )
        if incidence:
            grouped[incidence] = grouped.get(incidence, 0) + 1
        else:
            free += 1
        pending ^= bit

    adjacency = {
        index: set() for index in range(len(reduced_rows))
    }
    for incidence in grouped:
        first, *others = incidence
        for other in others:
            adjacency[first].add(other)
            adjacency[other].add(first)

    total = free
    unseen = set(adjacency)
    while unseen:
        first = min(unseen)
        stack = [first]
        unseen.remove(first)
        component_indices: list[int] = []
        while stack:
            current = stack.pop()
            component_indices.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        component_indices.sort()
        local_index = {
            original: index
            for index, original in enumerate(component_indices)
        }
        component_groups = [
            (
                tuple(local_index[index] for index in incidence),
                multiplicity,
            )
            for incidence, multiplicity in grouped.items()
            if incidence[0] in local_index
        ]
        component_groups.sort(
            key=lambda value: (-len(value[0]), value[1], value[0])
        )
        capacities = tuple(
            reduced_rows[index][1] for index in component_indices
        )
        group_count = len(component_groups)
        remaining_total = [0] * (group_count + 1)
        remaining_incident = [
            [0] * len(component_indices)
            for _ in range(group_count + 1)
        ]
        for position in range(group_count - 1, -1, -1):
            incidence, multiplicity = component_groups[position]
            remaining_total[position] = (
                remaining_total[position + 1] + multiplicity
            )
            remaining_incident[position] = list(
                remaining_incident[position + 1]
            )
            for index in incidence:
                remaining_incident[position][index] += multiplicity

        def normalize(
            position: int, residual: tuple[int, ...]
        ) -> tuple[int, ...]:
            return tuple(
                min(value, remaining_incident[position][index])
                for index, value in enumerate(residual)
            )

        @lru_cache(maxsize=None)
        def solve(
            position: int, residual: tuple[int, ...]
        ) -> int:
            residual = normalize(position, residual)
            if charge is not None:
                charge(
                    (
                        tuple(component_indices),
                        position,
                        residual,
                    )
                )
            if position == group_count:
                return 0
            incidence, multiplicity = component_groups[position]
            maximum = min(
                (multiplicity, *(residual[index] for index in incidence))
            )
            best = 0
            for selected in range(maximum, -1, -1):
                next_residual = list(residual)
                for index in incidence:
                    next_residual[index] -= selected
                candidate = selected + solve(
                    position + 1,
                    normalize(position + 1, tuple(next_residual)),
                )
                best = max(best, candidate)
                if best == remaining_total[position]:
                    break
            return best

        total += solve(0, normalize(0, capacities))
    return total


def _minimal_masks(values: Iterable[int]) -> tuple[int, ...]:
    """Return the inclusion-minimal masks in deterministic order."""

    kept: list[int] = []
    for value in sorted(set(values), key=lambda mask: (mask.bit_count(), mask)):
        if any((prior & value) == prior for prior in kept):
            continue
        kept.append(value)
    return tuple(kept)


def _large_component_bounds(
    component: tuple[str, ...],
    *,
    all_mask: int,
    family_index: dict[str, int],
    compiled_by_family: dict[str, list[_CompiledQuestion]],
    question_safe,
    budget: _StateBudget,
) -> _Bounds:
    """Solve a large monotone component through exact closure certificates.

    Forward-safe removals form a hereditary system.  In reverse, begin with
    the terminal families that remain and repeatedly restore any family that
    would have been safe immediately before its removal.  This is a monotone
    closure operator, and the reverse order of its additions is a valid
    forward witness.

    The maximum is the initially safe universe size minus the smallest seed
    that spans this closure.  A seed spans exactly when it intersects the
    complement of every proper closed set.  The cutting loop below solves the
    exact transversal of the closed-set complements seen so far, then either
    obtains a spanning seed or adds the complement of a maximal proper closed
    set as a violated cut.  The candidate is a lower bound until it spans, at
    which point the bound and concrete witness meet.

    For the minimum, a terminal consumed set must either contain each family
    or contain one exact disable mask for every variant of that family.  The
    branch-and-bound search adds those masks in batches and rejects a batch
    unless its complementary remaining set reverse-spans the universe.  That
    reachability test is exact, and infeasibility is hereditary, so pruning an
    unshellable batch also prunes all of its supersets.  Every cached closure,
    transversal, and terminal-search state consumes the same fail-closed
    state budget as the historical recurrence.
    """

    global_bits = tuple(1 << family_index[family_id] for family_id in component)
    def localize(mask: int) -> int:
        return sum(
            1 << index
            for index, bit in enumerate(global_bits)
            if mask & bit
        )

    initial_safe = tuple(
        index
        for index, family_id in enumerate(component)
        if any(
            question_safe(question, all_mask)
            for question in compiled_by_family[family_id]
        )
    )
    universe = sum(1 << index for index in initial_safe)
    if not universe:
        budget.consume(len(component))
        return _Bounds(0, 0, (), ())
    universe_global_mask = sum(global_bits[index] for index in initial_safe)

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
                failures.append(localize(without_question) & universe)
        union = (
            requirement.repair_mask | requirement.verification_mask
        ) & ~question_bit
        fixed_union = (union & ~universe_global_mask).bit_count()
        dynamic_union = localize(union) & universe
        if fixed_union == 0:
            pending = dynamic_union
            if not pending:
                failures.append(0)
            while pending:
                bit = pending & -pending
                pending ^= bit
                failures.append(dynamic_union ^ bit)
        elif fixed_union == 1:
            failures.append(dynamic_union)
        return _minimal_masks(failures)

    disable_masks: dict[int, tuple[int, ...]] = {}
    for index in initial_safe:
        family_id = component[index]
        combined: tuple[int, ...] = (0,)
        seen_variants: set[tuple[tuple[int, int], ...]] = set()
        for question in compiled_by_family[family_id]:
            signature = tuple(
                sorted(
                    (
                        requirement.repair_mask,
                        requirement.verification_mask,
                    )
                    for requirement in question.requirements
                )
            )
            if signature in seen_variants:
                continue
            seen_variants.add(signature)
            question_failures = _minimal_masks(
                failure
                for requirement in question.requirements
                for failure in requirement_failure_masks(question, requirement)
            )
            if not question_failures:
                combined = ()
                break
            combined = _minimal_masks(
                prior | failure
                for prior in combined
                for failure in question_failures
            )
        disable_masks[index] = combined

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
    equivalence: dict[tuple[object, ...], list[int]] = {}
    for index in initial_safe:
        family_id = component[index]
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
    equivalent_groups = tuple(tuple(indices) for indices in equivalence.values())
    group_masks = tuple(
        sum(1 << index for index in group) for group in equivalent_groups
    )
    group_prefixes = tuple(
        tuple(
            sum(1 << index for index in group[:count])
            for count in range(len(group) + 1)
        )
        for group in equivalent_groups
    )

    def canonicalize(consumed: int) -> int:
        return sum(
            prefixes[(consumed & group_mask).bit_count()]
            for group_mask, prefixes in zip(
                group_masks, group_prefixes, strict=True
            )
        )

    def family_safe(index: int, consumed: int) -> bool:
        return all(
            (failure & consumed) != failure
            for failure in disable_masks[index]
        )

    evaluated_safe_states: set[int] = set()

    @lru_cache(maxsize=None)
    def safe_mask(consumed: int) -> int:
        if consumed not in evaluated_safe_states:
            budget.consume(len(component))
            evaluated_safe_states.add(consumed)
        safe = 0
        pending = universe & ~consumed
        while pending:
            bit = pending & -pending
            pending ^= bit
            index = bit.bit_length() - 1
            if family_safe(index, consumed):
                safe |= bit
        return safe

    evaluated_closures: set[int] = set()

    @lru_cache(maxsize=None)
    def reverse_closure(seed: int) -> int:
        if seed not in evaluated_closures:
            budget.consume(len(component))
            evaluated_closures.add(seed)
        available = seed & universe
        while True:
            prior = available
            pending = universe & ~available
            while pending:
                bit = pending & -pending
                pending ^= bit
                index = bit.bit_length() - 1
                consumed = universe & ~(available | bit)
                if family_safe(index, consumed):
                    available |= bit
            if available == prior:
                return available

    def maximal_proper_closed(seed: int) -> int:
        closed = reverse_closure(seed)
        if closed == universe:
            raise AssertionError("A spanning seed has no proper closure cut.")
        pending = universe & ~closed
        while pending:
            bit = pending & -pending
            pending ^= bit
            extended = reverse_closure(closed | bit)
            if extended != universe:
                closed = extended
                pending &= ~closed
        return closed

    def normalize_cuts(values: Iterable[int]) -> tuple[int, ...]:
        return _minimal_masks(
            value & universe for value in values if value & universe
        )

    def lexicographic_mask(mask: int) -> tuple[str, ...]:
        return tuple(
            component[index]
            for index in range(len(component))
            if mask & (1 << index)
        )

    def minimum_transversal(cuts: tuple[int, ...]) -> int:
        rows = normalize_cuts(cuts)
        if not rows:
            budget.consume(len(component))
            return 0

        chosen = 0
        pending_rows = rows
        while pending_rows:
            bit = max(
                (1 << index for index in initial_safe),
                key=lambda candidate: (
                    sum(bool(row & candidate) for row in pending_rows),
                    -candidate.bit_length(),
                ),
            )
            chosen |= bit
            pending_rows = tuple(
                row for row in pending_rows if not row & bit
            )
        best = (chosen.bit_count(), lexicographic_mask(chosen), chosen)
        evaluated: dict[tuple[int, ...], tuple[int, tuple[str, ...]]] = {}

        def search(selected: int, remaining: tuple[int, ...]) -> None:
            nonlocal best
            remaining = normalize_cuts(remaining)
            prior = evaluated.get(remaining)
            selected_count = selected.bit_count()
            selected_key = (selected_count, lexicographic_mask(selected))
            if prior is not None and prior <= selected_key:
                return
            budget.consume(len(component))
            evaluated[remaining] = selected_key
            if selected_count > best[0]:
                return
            if not remaining:
                candidate = (
                    selected_count,
                    lexicographic_mask(selected),
                    selected,
                )
                if candidate < best:
                    best = candidate
                return
            disjoint: list[int] = []
            for row in sorted(
                remaining, key=lambda value: (value.bit_count(), value)
            ):
                if all(not row & prior_row for prior_row in disjoint):
                    disjoint.append(row)
            if selected_count + len(disjoint) > best[0]:
                return
            row = min(
                remaining,
                key=lambda value: (
                    value.bit_count(),
                    -sum(bool(value & other) for other in remaining),
                    value,
                ),
            )
            branches: list[tuple[int, str, int]] = []
            pending = row
            while pending:
                bit = pending & -pending
                pending ^= bit
                index = bit.bit_length() - 1
                branches.append(
                    (
                        -sum(bool(bit & other) for other in remaining),
                        component[index],
                        bit,
                    )
                )
            for _, _, bit in sorted(branches):
                search(
                    selected | bit,
                    tuple(other for other in remaining if not other & bit),
                )

        search(0, rows)
        return best[2]

    high_cuts: tuple[int, ...] = ()
    while True:
        terminal_seed = minimum_transversal(high_cuts)
        closed = reverse_closure(terminal_seed)
        if closed == universe:
            break
        maximal = maximal_proper_closed(closed)
        high_cuts = normalize_cuts((*high_cuts, universe & ~maximal))

    high = universe.bit_count() - terminal_seed.bit_count()

    def reverse_addition_order(seed: int) -> tuple[str, ...]:
        available = seed & universe
        additions: list[str] = []
        while available != universe:
            selected: int | None = None
            pending = universe & ~available
            while pending:
                bit = pending & -pending
                pending ^= bit
                index = bit.bit_length() - 1
                consumed = universe & ~(available | bit)
                if family_safe(index, consumed):
                    selected = index
                    break
            if selected is None:
                raise AssertionError("A spanning capacity seed lost its witness.")
            available |= 1 << selected
            additions.append(component[selected])
        additions.reverse()
        return tuple(additions)

    high_sequence = reverse_addition_order(terminal_seed)
    if len(high_sequence) != high:
        raise AssertionError("Maximum capacity witness length diverged from seed.")

    def minimum_terminal() -> int:
        consumed = 0
        while safe := safe_mask(consumed):
            candidates: list[tuple[int, str, int]] = []
            pending = safe
            while pending:
                bit = pending & -pending
                pending ^= bit
                index = bit.bit_length() - 1
                successor = canonicalize(consumed | bit)
                candidates.append(
                    (
                        safe_mask(successor).bit_count(),
                        component[index],
                        successor,
                    )
                )
            _, _, consumed = min(candidates)
        best = (consumed.bit_count(), lexicographic_mask(consumed), consumed)
        evaluated: set[int] = set()

        def search(state: int) -> None:
            nonlocal best
            state = canonicalize(state)
            if state in evaluated:
                return
            budget.consume(len(component))
            evaluated.add(state)
            depth = state.bit_count()
            if depth >= best[0]:
                return
            # A consumed batch is reachable exactly when its complementary
            # remaining seed reverse-spans the component.  Reachability is
            # hereditary, so an unshellable state has no shellable superset.
            if reverse_closure(universe & ~state) != universe:
                return
            safe = safe_mask(state)
            if not safe:
                candidate = (depth, lexicographic_mask(state), state)
                if candidate < best:
                    best = candidate
                return
            choices: list[tuple[int, int, str, tuple[int, ...]]] = []
            pending = safe
            while pending:
                bit = pending & -pending
                pending ^= bit
                index = bit.bit_length() - 1
                successor_values: set[int] = set()
                for failure in (bit, *disable_masks[index]):
                    successor = canonicalize(state | failure)
                    if (
                        successor != state
                        and successor.bit_count() < best[0]
                    ):
                        successor_values.add(successor)
                successors = tuple(
                    sorted(
                        successor_values,
                        key=lambda value: (
                            value.bit_count(),
                            value,
                        ),
                    )
                )
                if not successors:
                    return
                choices.append(
                    (
                        len(successors),
                        min(value.bit_count() - depth for value in successors),
                        component[index],
                        successors,
                    )
                )
            _, _, _, successors = min(choices)
            for successor in successors:
                search(successor)

        search(0)
        return best[2]

    low_terminal = minimum_terminal()
    remaining_seed = universe & ~low_terminal
    if reverse_closure(remaining_seed) != universe:
        raise AssertionError("Minimum terminal lost its reachability proof.")

    low = low_terminal.bit_count()
    # The reverse closure records a valid removal order for the terminal set.
    low_sequence = reverse_addition_order(remaining_seed)
    expected_low_families = {
        component[index]
        for index in range(len(component))
        if low_terminal & (1 << index)
    }
    if (
        len(low_sequence) != low
        or set(low_sequence) != expected_low_families
    ):
        raise AssertionError("Minimum capacity witness length diverged from terminal.")
    return _Bounds(low, high, low_sequence, high_sequence)


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

    # The historical sequence recurrence is ideal for small components and
    # preserves its stable lexicographic witnesses.  Large strongly connected
    # banks are solved by closure certificates, which quotient the otherwise
    # exponential enumeration of removal-order permutations.
    if len(component) >= _CLOSURE_SOLVER_MIN_COMPONENT_SIZE:
        return _large_component_bounds(
            component,
            all_mask=all_mask,
            family_index=family_index,
            compiled_by_family=compiled_by_family,
            question_safe=question_safe,
            budget=budget,
        )
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

    certified_high_bound: int | None = None
    certified_high_quota_blocks: tuple[tuple[int, int], ...] = ()
    certified_high_quota_candidates: tuple[tuple[int, int], ...] = ()

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

        nonlocal certified_high_bound
        nonlocal certified_high_quota_blocks
        nonlocal certified_high_quota_candidates
        initial_safe = safe_indices(0)
        charge_state(0)
        if not initial_safe:
            certified_high_bound = 0
            certified_high_quota_blocks = ()
            certified_high_quota_candidates = ()
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
        certified_high_quota_candidates = quota_candidates
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
            # This exact-cover oracle is exponential too.  Charge each cached
            # subproblem just like the sequence and quota recurrences so a
            # low --state-limit always fails closed in bounded work.
            budget.consume(len(component))
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
        certified_high_bound = partition_cost
        certified_high_quota_blocks = quota_blocks

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

    def certified_low_bound() -> tuple[int, tuple[str, ...]] | None:
        """Certify the earliest terminal path through an order-free relaxation.

        A real removal sequence can end only at a consumed-family set that
        disables every initially safe family.  Ignore removal order and derive
        the exact minimal masks that can disable each family: either consume
        the family itself, or exhaust enough repair/verification reserves to
        make every one of its question variants unsafe.

        If no such terminal set is smaller than a real lexicographic greedy
        path, the greedy path is an exact earliest-exhaustion witness.  The
        relaxation admits states that no real order can reach, so failure to
        certify falls back to exhaustive search; it can never overstate the
        lower bound.
        """

        initial_safe = safe_indices(0)
        charge_state(0)
        if not initial_safe:
            return 0, ()
        universe_mask = sum(1 << index for index in initial_safe)
        universe_global_mask = sum(
            global_bits[index] for index in initial_safe
        )

        def localize(mask: int) -> int:
            return sum(
                1 << index
                for index, bit in enumerate(global_bits)
                if mask & bit
            )

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

        # This is a real sequence, not a relaxation. Prefer the removal that
        # leaves the fewest safe successors, then the stable family ID.  It is
        # the same deterministic first branch as the exhaustive lower-bound
        # recurrence and commonly exposes a small cut set immediately.
        consumed = 0
        greedy_sequence: list[str] = []
        while True:
            charge_state(consumed)
            safe = safe_indices(consumed)
            if not safe:
                break
            selected = min(
                branch_indices(safe),
                key=lambda index: (
                    len(
                        safe_indices(
                            canonicalize(consumed | (1 << index))
                        )
                    ),
                    component[index],
                ),
            )
            greedy_sequence.append(component[selected])
            consumed = canonicalize(consumed | (1 << selected))
        candidate_bound = len(greedy_sequence)
        if candidate_bound == 0:
            return 0, ()

        terminal_solvers: dict[int, Callable[[int], bool]] = {}

        def terminal_solver(maximum_consumed: int):
            cached = terminal_solvers.get(maximum_consumed)
            if cached is not None:
                return cached

            @lru_cache(maxsize=None)
            def terminal_exists(state: int) -> bool:
                if state.bit_count() > maximum_consumed:
                    return False
                charge_state(state)
                safe = safe_indices(state)
                if not safe:
                    return True
                choices: list[tuple[int, str, tuple[int, ...]]] = []
                for index in branch_indices(safe):
                    next_states = {
                        canonicalize(state | failure)
                        for failure in disable_masks[index]
                        if failure & ~state
                    }
                    next_states = {
                        next_state
                        for next_state in next_states
                        if next_state != state
                        and next_state.bit_count() <= maximum_consumed
                    }
                    # Every terminal superset must disable this currently safe
                    # family. If even the order-free relaxation cannot do so
                    # inside the bound, no terminal set exists below this node.
                    if not next_states:
                        return False
                    choices.append(
                        (
                            len(next_states),
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
                _, _, next_states = min(choices)
                return any(
                    terminal_exists(next_state)
                    for next_state in next_states
                )

            terminal_solvers[maximum_consumed] = terminal_exists
            return terminal_exists

        # Find the exact minimum order-free terminal cardinality. This is a
        # lower bound on every real sequence because it permits families to be
        # consumed in an impossible batch. The real greedy terminal supplies a
        # finite upper bound, so binary search is complete.
        lower = 0
        upper = candidate_bound
        while lower < upper:
            middle = (lower + upper) // 2
            if terminal_solver(middle)(0):
                upper = middle
            else:
                lower = middle + 1
        relaxed_low = lower

        @lru_cache(maxsize=None)
        def reverse_shellable(terminal: int, required_prefix: int) -> bool:
            """Whether a terminal set has a real order after one fixed prefix.

            In reverse, remove a last-consumed family from ``terminal`` only
            when it was safe just before that removal. Adding reserves is
            monotone: once a family is a valid reverse candidate, removing a
            different valid candidate cannot make it invalid. Consequently
            any available reverse candidate is safe to choose and existence
            needs no permutation backtracking.
            """

            if required_prefix & ~terminal:
                return False
            current = terminal
            while current != required_prefix:
                candidates = []
                pending = current & ~required_prefix
                while pending:
                    bit = pending & -pending
                    index = bit.bit_length() - 1
                    before_last = current ^ bit
                    if index in safe_indices(before_last):
                        candidates.append(index)
                    pending ^= bit
                if not candidates:
                    return False
                # Existence is choice-independent by monotonicity. Stable
                # ordering keeps diagnostic behavior deterministic.
                selected = min(candidates, key=lambda value: component[value])
                current ^= 1 << selected
            return True

        def shellable_terminal(
            required_prefix: int, target_depth: int
        ) -> int | None:
            """Find any exact-depth terminal set extending a real prefix."""

            terminal_possible = terminal_solver(target_depth)

            @lru_cache(maxsize=None)
            def search(state: int) -> int | None:
                if required_prefix & ~state:
                    raise AssertionError(
                        "Terminal relaxation discarded a required prefix."
                    )
                charge_state(state)
                # Reachable consumed sets are hereditary: restricting any real
                # removal order to a subset only restores reserves, so every
                # retained removal remains safe. Therefore an unshellable batch
                # state cannot be repaired by consuming still more families.
                if (
                    not reverse_shellable(state, required_prefix)
                    or not terminal_possible(state)
                ):
                    return None
                safe = safe_indices(state)
                if not safe:
                    if (
                        state.bit_count() == target_depth
                        and reverse_shellable(state, required_prefix)
                    ):
                        return state
                    return None
                choices: list[
                    tuple[int, str, tuple[int, ...]]
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
                        if next_state != state
                        and next_state.bit_count() <= target_depth
                        and not required_prefix & ~next_state
                    }
                    if not next_states:
                        return None
                    choices.append(
                        (
                            len(next_states),
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
                _, _, next_states = min(choices)
                for next_state in next_states:
                    found = search(next_state)
                    if found is not None:
                        return found
                return None

            return search(required_prefix)

        # Search real paths from the proven order-free lower bound upward.
        # At each prefix, ask the terminal-set solver whether a shellable
        # completion exists before accepting the smallest safe family. This
        # recovers the same lexicographic witness as exhaustive sequence search
        # without enumerating permutations of one terminal set.
        def real_terminal_sequence(
            target_depth: int,
        ) -> tuple[str, ...] | None:
            if shellable_terminal(0, target_depth) is None:
                return None

            sequence: list[str] = []
            state = 0
            while True:
                safe = safe_indices(state)
                if not safe:
                    break
                selected: tuple[str, int] | None = None
                for index in sorted(
                    branch_indices(safe), key=lambda value: component[value]
                ):
                    next_state = canonicalize(state | (1 << index))
                    if shellable_terminal(
                        next_state, target_depth
                    ) is not None:
                        selected = component[index], next_state
                        break
                if selected is None:
                    raise AssertionError(
                        "Reachable capacity terminal lost its witness path."
                    )
                family_id, state = selected
                sequence.append(family_id)
            if len(sequence) != target_depth:
                raise AssertionError(
                    "Capacity witness length diverged from certified depth."
                )
            return tuple(sequence)

        for depth in range(relaxed_low, candidate_bound + 1):
            sequence = real_terminal_sequence(depth)
            if sequence is not None:
                return depth, sequence
        raise AssertionError(
            "A real greedy capacity terminal escaped bounded reconstruction."
        )

    low_certificate = certified_low_bound()

    def certified_high_sequence() -> tuple[int, tuple[str, ...]] | None:
        """Prove the latest terminal path below a certified packing bound.

        ``certified_equal_bound`` derives a disjoint cover of the initially
        safe families with mandatory consumption quotas.  Even when its first
        greedy witness does not attain the resulting upper bound, those quotas
        remain valid.  Search candidate depths from that bound downward and
        decide whether a real prefix of each depth exists.  The first reachable
        depth is exact: every larger depth was disproved and the packing
        certificate excludes anything beyond the initial bound.

        At an intermediate state, safety is monotone under further removals.
        A family that is unsafe now can never be consumed later. Restrict every
        valid mandatory quota to the currently safe families, subtract quota
        already consumed, and solve the resulting grouped 0/1 cardinality
        packing exactly. Its optimum is a sound state-specific upper bound on
        real removals and accounts for all overlapping quotas simultaneously.
        """

        if certified_high_bound is None:
            return None
        if certified_high_bound == 0:
            return 0, ()
        if (
            not certified_high_quota_blocks
            or not certified_high_quota_candidates
        ):
            return None

        def within_quotas(state: int) -> bool:
            return all(
                (state & block).bit_count() <= quota
                for block, quota in certified_high_quota_candidates
            )

        def dynamic_quota_constraints(
            state: int, safe_mask: int
        ) -> tuple[tuple[int, int], ...]:
            by_block: dict[int, int] = {}
            for block, quota in certified_high_quota_candidates:
                restricted = block & safe_mask
                if not restricted:
                    continue
                residual = quota - (state & block).bit_count()
                if residual < 0:
                    return ()
                residual = min(residual, restricted.bit_count())
                by_block[restricted] = min(
                    by_block.get(restricted, restricted.bit_count()),
                    residual,
                )
            return tuple(sorted(by_block.items()))

        evaluated_quota_states: set[tuple[object, ...]] = set()

        def charge_quota_state(
            state: int, oracle: tuple[object, ...]
        ) -> None:
            oracle_state = (state, *oracle)
            if oracle_state not in evaluated_quota_states:
                budget.consume(len(component))
                evaluated_quota_states.add(oracle_state)

        @lru_cache(maxsize=None)
        def quota_upper(state: int) -> int:
            if not within_quotas(state):
                return -1
            safe_mask = sum(
                1 << index for index in safe_indices(state)
            )
            future = _maximum_quota_packing(
                safe_mask,
                dynamic_quota_constraints(state, safe_mask),
                charge=lambda oracle: charge_quota_state(state, oracle),
            )
            if future < 0:
                return -1
            return state.bit_count() + future

        # Establish a real lower witness so target search never descends below
        # a depth already known to be attainable.
        greedy_state = 0
        greedy_sequence: list[str] = []
        while True:
            charge_state(greedy_state)
            safe = safe_indices(greedy_state)
            if not safe:
                break
            selected = min(
                branch_indices(safe),
                key=lambda index: (
                    -len(
                        safe_indices(greedy_state | (1 << index))
                    ),
                    component[index],
                ),
            )
            greedy_sequence.append(component[selected])
            greedy_state |= 1 << selected
        greedy_bound = len(greedy_sequence)
        effective_high_bound = min(
            certified_high_bound, quota_upper(0)
        )
        if effective_high_bound < greedy_bound:
            raise AssertionError(
                "A valid capacity witness exceeded its quota certificate."
            )

        # A bounded beam is only an attainment accelerator. Missing the upper
        # bound proves nothing and falls through to exact target search. If it
        # reaches the certified packing bound, however, that concrete safe
        # sequence is a valid exact witness. Keep all generated states inside
        # the same fail-closed budget as the proof searches.
        beam: dict[int, tuple[str, ...]] = {0: ()}
        attainment_sequence: tuple[str, ...] | None = None
        for _ in range(effective_high_bound):
            candidates: dict[int, tuple[str, ...]] = {}
            for state, sequence in sorted(
                beam.items(), key=lambda value: value[1]
            ):
                for index in branch_indices(safe_indices(state)):
                    next_state = state | (1 << index)
                    if not within_quotas(next_state):
                        continue
                    charge_state(next_state)
                    next_sequence = sequence + (component[index],)
                    prior = candidates.get(next_state)
                    if prior is None or next_sequence < prior:
                        candidates[next_state] = next_sequence
            if not candidates:
                break
            if next(iter(candidates)).bit_count() == effective_high_bound:
                terminal = [
                    sequence
                    for state, sequence in candidates.items()
                    if not safe_indices(state)
                ]
                if terminal:
                    attainment_sequence = min(terminal)
                break

            def beam_rank(
                item: tuple[int, tuple[str, ...]],
            ) -> tuple[int, int, int, tuple[str, ...], int]:
                state, sequence = item
                safe_count = len(safe_indices(state))
                slack_damage = sum(
                    (state & block).bit_count()
                    * max(1, block.bit_count() - quota)
                    for block, quota in certified_high_quota_candidates
                    if quota < block.bit_count()
                )
                return (
                    -quota_upper(state),
                    -safe_count,
                    slack_damage,
                    sequence,
                    state,
                )

            beam = dict(
                sorted(candidates.items(), key=beam_rank)[:256]
            )

        attaining_prefixes: set[int] = set()
        if attainment_sequence is not None:
            state = 0
            attaining_prefixes.add(state)
            by_family = {
                family_id: index
                for index, family_id in enumerate(component)
            }
            for family_id in attainment_sequence:
                state |= 1 << by_family[family_id]
                attaining_prefixes.add(state)

        target_depths: Iterable[int]
        if attainment_sequence is not None:
            target_depths = (effective_high_bound,)
        else:
            target_depths = range(
                effective_high_bound, greedy_bound - 1, -1
            )

        for target_depth in target_depths:

            @lru_cache(maxsize=None)
            def reaches_target(state: int) -> bool:
                charge_state(state)
                if state in attaining_prefixes:
                    return True
                depth = state.bit_count()
                if depth == target_depth:
                    return True
                safe = safe_indices(state)
                if not safe or quota_upper(state) < target_depth:
                    return False
                children: list[
                    tuple[int, int, str, int]
                ] = []
                for index in branch_indices(safe):
                    next_state = state | (1 << index)
                    if not within_quotas(next_state):
                        continue
                    next_safe = safe_indices(next_state)
                    next_upper = quota_upper(next_state)
                    if next_upper < target_depth:
                        continue
                    children.append(
                        (
                            -next_upper,
                            -len(next_safe),
                            component[index],
                            next_state,
                        )
                    )
                return any(
                    reaches_target(next_state)
                    for _, _, _, next_state in sorted(children)
                )

            if not reaches_target(0):
                continue

            # Recover the same stable lexicographic witness as the exhaustive
            # recurrence by choosing the first concrete family whose suffix
            # can still attain the proven maximum.
            sequence: list[str] = []
            state = 0
            while state.bit_count() < target_depth:
                selected: tuple[str, int] | None = None
                for index in sorted(
                    branch_indices(safe_indices(state)),
                    key=lambda value: component[value],
                ):
                    next_state = state | (1 << index)
                    if (
                        within_quotas(next_state)
                        and reaches_target(next_state)
                    ):
                        selected = component[index], next_state
                        break
                if selected is None:
                    raise AssertionError(
                        "Reachable maximum capacity lost its witness path."
                    )
                family_id, state = selected
                sequence.append(family_id)
            if safe_indices(state):
                raise AssertionError(
                    "Certified maximum capacity ended before exhaustion."
                )
            return target_depth, tuple(sequence)

        # The real greedy path above must be admitted by every sound quota.
        # Preserve the historical exhaustive recurrence as a fail-closed
        # fallback if an internal certificate invariant is ever violated.
        return None

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

    if low_certificate is None:
        low, low_sequence = lowest(0)
    else:
        low, low_sequence = low_certificate
    high_certificate = certified_high_sequence()
    if high_certificate is None:
        high, high_sequence = highest(0)
    else:
        high, high_sequence = high_certificate
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
