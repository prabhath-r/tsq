#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0

"""Falsify declared question-family independence with transparent evidence.

The production selector treats different ``family_id`` values as independent
opportunities for diagnosis, repair, and verification.  This laboratory asks a
narrower, adversarial question: which active cross-family items look similar
enough that a human reviewer should verify that they require genuinely
different solution paths?

Token overlap is deliberately used only to nominate candidates.  It is not a
semantic model and cannot establish dependence.  For every declared or
signal-nominated cluster, the lab also stress-tests the exact sustained-capacity
analyzer by counterfactually treating the cluster as one family.  A capacity
drop means independence is operationally important *if* a reviewer later
confirms dependence; it does not confirm dependence itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tsq.adaptive import BOUNDARY_ALGORITHM_VERSION  # noqa: E402
from tsq.capacity import (  # noqa: E402
    ALGORITHM_VERSION as CAPACITY_ALGORITHM_VERSION,
    CapacityTarget,
    TargetCapacity,
    analyze_sustained_capacity,
)
from tsq.corpus import load_bundle, read_and_parse  # noqa: E402
from tsq.graph import KnowledgeGraph  # noqa: E402
from tsq.models import LearningObjective, Question  # noqa: E402
from tsq.policy import POLICY_VERSION  # noqa: E402
from tsq.versions import DEFAULT_LEARNER_MODEL_VERSION  # noqa: E402


LAB_VERSION = "family-independence-falsification-v1"
NORMALIZATION_VERSION = "lower-alnum-stopwords-v1"
DEFAULT_CORPUS = PROJECT_ROOT / "corpus" / "ai_curriculum.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "experiments" / "results" / "family_independence_lab.json"
)
STEM_OVERLAP_THRESHOLD = 0.27
SOLUTION_OVERLAP_THRESHOLD = 0.37
COMBINED_JACCARD_THRESHOLD = 0.29

# These are review declarations, not declarations of dependence.  They make
# known, high-consequence cases durable even if a future wording edit pushes a
# lexical score just below the automatic nomination threshold.
DECLARED_REVIEW_CLUSTERS: tuple[dict[str, object], ...] = (
    {
        "id": "attention_value_routing_trio",
        "question_ids": (
            "q_attention_value_role_ablation_001",
            "q_attention_value_projection_counterfactual_001",
            "q_attention_value_perturbation_001",
        ),
        "review_reason": (
            "All three items use the same fine objective and named-error routes "
            "while varying a value-vector perturbation. A reviewer must decide "
            "whether the required traces are independently diagnostic."
        ),
    },
    {
        "id": "multiquery_kv_cache_trio",
        "question_ids": (
            "q_transformer_multiquery_cache_axes_001",
            "q_transformer_multiquery_cache_inventory_001",
            "q_transformer_multiquery_state_transition_001",
        ),
        "review_reason": (
            "All three items use the same fine objective and named-error routes "
            "while tracing multi-query key-value cache state. A reviewer must "
            "decide whether the tensor-accounting operations are independent."
        ),
    },
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "all",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "because",
        "before",
        "but",
        "by",
        "can",
        "does",
        "each",
        "for",
        "from",
        "has",
        "have",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "not",
        "of",
        "on",
        "one",
        "only",
        "or",
        "so",
        "than",
        "that",
        "the",
        "their",
        "then",
        "this",
        "to",
        "what",
        "when",
        "which",
        "while",
        "with",
    }
)


class LabInvariantError(RuntimeError):
    """Raised when the laboratory cannot make its conservative comparison."""


@dataclass(frozen=True, slots=True)
class QuestionFeatures:
    question: Question
    stem_tokens: frozenset[str]
    solution_tokens: frozenset[str]

    @property
    def misconception_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.question.misconception_ids))


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalized_tokens(text: str) -> frozenset[str]:
    """Expose the exact, intentionally simple lexical representation."""

    return frozenset(
        token
        for token in _TOKEN_PATTERN.findall(text.casefold())
        if token not in _STOPWORDS
    )


def question_features(question: Question) -> QuestionFeatures:
    # Correct text plus every rationale captures the authored solution and the
    # distinctions used to reject named misconceptions. Distractor surface text
    # is excluded so a repeated misconception label is not counted twice.
    solution_text = " ".join(
        (
            question.correct_option.text,
            *(option.rationale for option in question.options),
        )
    )
    return QuestionFeatures(
        question=question,
        stem_tokens=normalized_tokens(question.stem),
        solution_tokens=normalized_tokens(solution_text),
    )


def _overlap(left: frozenset[str], right: frozenset[str]) -> dict[str, object]:
    shared = left & right
    union = left | right
    minimum = min(len(left), len(right))
    return {
        "left_token_count": len(left),
        "right_token_count": len(right),
        "intersection_count": len(shared),
        "union_count": len(union),
        "jaccard": round(len(shared) / len(union), 6) if union else 1.0,
        "overlap_coefficient": (
            round(len(shared) / minimum, 6) if minimum else 1.0
        ),
        "shared_tokens": sorted(shared),
    }


def pair_evidence(
    left: QuestionFeatures, right: QuestionFeatures
) -> dict[str, object]:
    if left.question.id == right.question.id:
        raise ValueError("Pair evidence requires two different questions.")
    stem = _overlap(left.stem_tokens, right.stem_tokens)
    solution = _overlap(left.solution_tokens, right.solution_tokens)
    combined = _overlap(
        left.stem_tokens | left.solution_tokens,
        right.stem_tokens | right.solution_tokens,
    )
    same_objective = (
        left.question.objective_id is not None
        and left.question.objective_id == right.question.objective_id
    )
    same_misconceptions = (
        bool(left.misconception_ids)
        and left.misconception_ids == right.misconception_ids
    )
    qualifies = bool(
        same_objective
        and same_misconceptions
        and (
            (
                stem["overlap_coefficient"] >= STEM_OVERLAP_THRESHOLD
                and solution["overlap_coefficient"]
                >= SOLUTION_OVERLAP_THRESHOLD
            )
            or combined["jaccard"] >= COMBINED_JACCARD_THRESHOLD
        )
    )
    return {
        "left_question_id": min(left.question.id, right.question.id),
        "right_question_id": max(left.question.id, right.question.id),
        "left_family_id": (
            left.question.family_id
            if left.question.id < right.question.id
            else right.question.family_id
        ),
        "right_family_id": (
            right.question.family_id
            if left.question.id < right.question.id
            else left.question.family_id
        ),
        "objective_id": left.question.objective_id if same_objective else None,
        "identical_named_misconception_set": same_misconceptions,
        "named_misconception_ids": (
            list(left.misconception_ids) if same_misconceptions else []
        ),
        "stem_overlap": stem,
        "solution_path_overlap": solution,
        "combined_overlap": combined,
        "signal_threshold_passed": qualifies,
        "candidate_status": (
            "requires_independent_semantic_review"
            if qualifies
            else "below_automatic_nomination_threshold"
        ),
        "semantic_dependence_established": False,
    }


def _connected_components(
    question_ids: Iterable[str],
    edges: Iterable[tuple[str, str]],
) -> tuple[tuple[str, ...], ...]:
    adjacency = {question_id: set() for question_id in question_ids}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    components: list[tuple[str, ...]] = []
    seen: set[str] = set()
    for question_id in sorted(adjacency):
        if question_id in seen or not adjacency[question_id]:
            continue
        pending = [question_id]
        seen.add(question_id)
        component: list[str] = []
        while pending:
            current = pending.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor not in seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(sorted(components))


def _capacity_snapshot(value: TargetCapacity) -> dict[str, object]:
    return {
        "exact": True,
        "capacity_algorithm": CAPACITY_ALGORITHM_VERSION,
        "eligible_questions": value.eligible_question_count,
        "eligible_families": value.eligible_family_count,
        "available_scope_families": value.available_scope_family_count,
        "initial_safe_family_count": len(value.initial_safe_family_ids),
        "initial_safe_family_ids": list(value.initial_safe_family_ids),
        "order_robust_main_capacity": value.order_robust_main_capacity,
        "achievable_main_capacity": value.achievable_main_capacity,
        "target_main_count": value.target_main_count,
        "aggregate_status": value.aggregate_status,
        "conservative_owned_concept_status": value.status,
        "states_evaluated": value.states_evaluated,
    }


def _objective_capacity(
    *,
    objective: LearningObjective,
    misconception_ids: tuple[str, ...],
    questions: Sequence[Question],
    graph: KnowledgeGraph,
    misconceptions: Sequence[object],
    unavailable_family_ids: Iterable[str] = (),
) -> TargetCapacity:
    objective_questions = tuple(
        question
        for question in questions
        if question.objective_id == objective.id
        and tuple(sorted(question.misconception_ids)) == misconception_ids
        and question.status.eligible_for_adaptation
    )
    owned_concepts = tuple(
        sorted({question.primary_concept_id for question in objective_questions})
    )
    if not objective_questions or not owned_concepts:
        raise LabInvariantError(
            f"Objective {objective.id} has no active questions to analyze."
        )
    target = CapacityTarget(
        target_id=f"objective:{objective.id}",
        target_type="topic",
        owned_concept_ids=owned_concepts,
        target_main_count=3,
    )
    report = analyze_sustained_capacity(
        objective_questions,
        graph,
        misconceptions,
        (target,),
        unavailable_family_ids=unavailable_family_ids,
    )
    return report.targets[0]


def _capacity_comparison(
    *,
    question_ids: Sequence[str],
    by_question_id: dict[str, Question],
    objectives: dict[str, LearningObjective],
    questions: Sequence[Question],
    graph: KnowledgeGraph,
    misconceptions: Sequence[object],
    baseline_cache: dict[str, TargetCapacity],
) -> dict[str, object]:
    selected = [by_question_id[question_id] for question_id in question_ids]
    objective_ids = {question.objective_id for question in selected}
    family_ids = tuple(sorted({question.family_id for question in selected}))
    if len(objective_ids) != 1 or None in objective_ids:
        raise LabInvariantError(
            "A counterfactual cluster must share one fine objective."
        )
    if len(family_ids) < 2:
        raise LabInvariantError(
            "A counterfactual cluster requires at least two declared families."
        )
    objective_id = next(iter(objective_ids))
    objective = objectives[objective_id]
    misconception_ids = tuple(sorted(selected[0].misconception_ids))
    cache_id = f"{objective_id}|{'|'.join(misconception_ids)}"
    baseline = baseline_cache.get(cache_id)
    if baseline is None:
        baseline = _objective_capacity(
            objective=objective,
            misconception_ids=misconception_ids,
            questions=questions,
            graph=graph,
            misconceptions=misconceptions,
        )
        baseline_cache[cache_id] = baseline
    retained_family_id = family_ids[0]
    collapsed_family_ids = family_ids[1:]
    collapsed = _objective_capacity(
        objective=objective,
        misconception_ids=misconception_ids,
        questions=questions,
        graph=graph,
        misconceptions=misconceptions,
        unavailable_family_ids=collapsed_family_ids,
    )
    robust_drop = (
        baseline.order_robust_main_capacity
        - collapsed.order_robust_main_capacity
    )
    achievable_drop = (
        baseline.achievable_main_capacity - collapsed.achievable_main_capacity
    )
    safe_drop = len(baseline.initial_safe_family_ids) - len(
        collapsed.initial_safe_family_ids
    )
    critical = robust_drop > 0 or achievable_drop > 0 or safe_drop > 0
    return {
        "scope": (
            "active_questions_with_same_fine_objective_and_identical_named_"
            "misconception_set"
        ),
        "objective_id": objective_id,
        "objective_name": objective.name,
        "counterfactual_assumption": (
            "Treat the nominated families as one semantic family by retaining "
            "one deterministic representative and removing the others."
        ),
        "retained_representative_family_id": retained_family_id,
        "counterfactually_unavailable_family_ids": list(collapsed_family_ids),
        "baseline": _capacity_snapshot(baseline),
        "collapsed": _capacity_snapshot(collapsed),
        "impact": {
            "initial_safe_family_drop": safe_drop,
            "order_robust_main_capacity_drop": robust_drop,
            "achievable_main_capacity_drop": achievable_drop,
            "capacity_critical_if_dependence_confirmed": critical,
            "assessment": (
                "capacity_critical_independence_candidate"
                if critical
                else "no_capacity_change_in_this_exact_scope"
            ),
        },
    }


def _cluster_pair_evidence(
    question_ids: Sequence[str],
    features: dict[str, QuestionFeatures],
) -> list[dict[str, object]]:
    pairs: list[dict[str, object]] = []
    for index, left_id in enumerate(question_ids):
        for right_id in question_ids[index + 1 :]:
            pairs.append(pair_evidence(features[left_id], features[right_id]))
    return sorted(
        pairs,
        key=lambda row: (
            row["left_question_id"],
            row["right_question_id"],
        ),
    )


def build_report(corpus: Path = DEFAULT_CORPUS) -> dict[str, object]:
    raw = load_bundle(corpus)
    (
        concepts,
        edges,
        misconceptions,
        _sources,
        questions,
        _domains,
        _topics,
    ) = read_and_parse(corpus, include_catalog=True)
    active = tuple(
        sorted(
            (
                question
                for question in questions
                if question.status.eligible_for_adaptation
            ),
            key=lambda question: question.id,
        )
    )
    by_question_id = {question.id: question for question in active}
    features = {
        question.id: question_features(question) for question in active
    }
    objectives = {
        question.objective.id: question.objective
        for question in active
        if question.objective is not None
    }
    graph = KnowledgeGraph(concepts, edges)

    eligible_groups: dict[
        tuple[str, tuple[str, ...]], list[QuestionFeatures]
    ] = {}
    for value in features.values():
        signature = (value.question.objective_id, value.misconception_ids)
        if signature[0] is None or not signature[1]:
            continue
        eligible_groups.setdefault(signature, []).append(value)

    all_pairs: list[dict[str, object]] = []
    qualifying_edges: list[tuple[str, str]] = []
    for group in eligible_groups.values():
        ordered = sorted(group, key=lambda value: value.question.id)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                if left.question.family_id == right.question.family_id:
                    continue
                evidence = pair_evidence(left, right)
                all_pairs.append(evidence)
                if evidence["signal_threshold_passed"]:
                    qualifying_edges.append(
                        (left.question.id, right.question.id)
                    )
    duplicate_candidates = sorted(
        (
            row for row in all_pairs if row["signal_threshold_passed"]
        ),
        key=lambda row: (
            row["left_question_id"],
            row["right_question_id"],
        ),
    )
    signal_components = _connected_components(
        features, qualifying_edges
    )

    baseline_cache: dict[str, TargetCapacity] = {}
    declared_results: list[dict[str, object]] = []
    for declaration in DECLARED_REVIEW_CLUSTERS:
        question_ids = tuple(declaration["question_ids"])
        missing = sorted(set(question_ids) - set(by_question_id))
        if missing:
            raise LabInvariantError(
                f"Declared review cluster {declaration['id']} has inactive or "
                f"missing questions: {', '.join(missing)}"
            )
        selected = [by_question_id[question_id] for question_id in question_ids]
        signatures = {
            (
                question.objective_id,
                tuple(sorted(question.misconception_ids)),
            )
            for question in selected
        }
        if len(signatures) != 1 or not next(iter(signatures))[1]:
            raise LabInvariantError(
                f"Declared review cluster {declaration['id']} no longer shares "
                "one objective and one non-empty named-misconception set."
            )
        pair_rows = _cluster_pair_evidence(question_ids, features)
        declared_results.append(
            {
                "cluster_id": declaration["id"],
                "origin": "declared_review_seed",
                "question_ids": list(question_ids),
                "family_ids": sorted(
                    question.family_id for question in selected
                ),
                "objective_id": selected[0].objective_id,
                "named_misconception_ids": sorted(
                    selected[0].misconception_ids
                ),
                "review_reason": declaration["review_reason"],
                "review_status": "independent_semantic_review_required",
                "semantic_dependence_established": False,
                "automatic_signal_connected": _connected_components(
                    question_ids,
                    (
                        (
                            row["left_question_id"],
                            row["right_question_id"],
                        )
                        for row in pair_rows
                        if row["signal_threshold_passed"]
                    ),
                )
                == (tuple(sorted(question_ids)),),
                "pair_evidence": pair_rows,
                "capacity_stress_test": _capacity_comparison(
                    question_ids=question_ids,
                    by_question_id=by_question_id,
                    objectives=objectives,
                    questions=active,
                    graph=graph,
                    misconceptions=misconceptions,
                    baseline_cache=baseline_cache,
                ),
            }
        )

    declared_sets = {
        frozenset(result["question_ids"]) for result in declared_results
    }
    suspected_results: list[dict[str, object]] = []
    for question_ids in signal_components:
        if frozenset(question_ids) in declared_sets:
            continue
        selected = [by_question_id[question_id] for question_id in question_ids]
        # A component cannot cross signatures because qualifying edges require
        # an exact signature match, but keep the invariant explicit.
        if len({question.objective_id for question in selected}) != 1:
            raise LabInvariantError("Signal component crossed objectives.")
        suspected_results.append(
            {
                "cluster_id": (
                    f"signal_cluster_{len(suspected_results) + 1:03d}"
                ),
                "origin": "transparent_lexical_signal",
                "question_ids": list(question_ids),
                "family_ids": sorted(
                    question.family_id for question in selected
                ),
                "objective_id": selected[0].objective_id,
                "named_misconception_ids": sorted(
                    selected[0].misconception_ids
                ),
                "review_status": "independent_semantic_review_required",
                "semantic_dependence_established": False,
                "pair_evidence": _cluster_pair_evidence(
                    question_ids, features
                ),
                "capacity_stress_test": _capacity_comparison(
                    question_ids=question_ids,
                    by_question_id=by_question_id,
                    objectives=objectives,
                    questions=active,
                    graph=graph,
                    misconceptions=misconceptions,
                    baseline_cache=baseline_cache,
                ),
            }
        )

    clusters = [*declared_results, *suspected_results]
    critical_ids = [
        cluster["cluster_id"]
        for cluster in clusters
        if cluster["capacity_stress_test"]["impact"][
            "capacity_critical_if_dependence_confirmed"
        ]
    ]
    deterministic: dict[str, object] = {
        "lab_version": LAB_VERSION,
        "corpus": {
            "path": (
                str(corpus.resolve().relative_to(PROJECT_ROOT))
                if corpus.resolve().is_relative_to(PROJECT_ROOT)
                else str(corpus)
            ),
            "sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
            "schema_version": raw["schema_version"],
            "title": raw["title"],
            "active_question_count": len(active),
            "active_family_count": len(
                {question.family_id for question in active}
            ),
        },
        "production_versions": {
            "selection_policy": POLICY_VERSION,
            "learner_model": DEFAULT_LEARNER_MODEL_VERSION,
            "selection_boundary": BOUNDARY_ALGORITHM_VERSION,
            "capacity_algorithm": CAPACITY_ALGORITHM_VERSION,
        },
        "analysis_contract": {
            "normalization_version": NORMALIZATION_VERSION,
            "token_pattern": _TOKEN_PATTERN.pattern,
            "stopwords": sorted(_STOPWORDS),
            "candidate_requires": (
                "same non-null fine objective, identical non-empty named "
                "misconception set, different declared families, and the "
                "documented lexical threshold"
            ),
            "thresholds": {
                "stem_overlap_coefficient": STEM_OVERLAP_THRESHOLD,
                "solution_path_overlap_coefficient": (
                    SOLUTION_OVERLAP_THRESHOLD
                ),
                "combined_jaccard_alternative": (
                    COMBINED_JACCARD_THRESHOLD
                ),
            },
            "solution_path_text": (
                "correct option text plus every authored option rationale"
            ),
            "lexical_signal_semantics": (
                "candidate nomination only; lexical overlap does not prove "
                "semantic dependence or interchangeability"
            ),
            "counterfactual_semantics": (
                "capacity consequence if a nominated cluster were one family; "
                "the collapse is not a corpus mutation"
            ),
        },
        "duplicate_pair_candidates": duplicate_candidates,
        "declared_review_clusters": declared_results,
        "signal_nominated_clusters": suspected_results,
        "findings": {
            "eligible_signature_pair_count": len(all_pairs),
            "duplicate_pair_candidate_count": len(duplicate_candidates),
            "declared_review_cluster_count": len(declared_results),
            "signal_nominated_cluster_count": len(suspected_results),
            "capacity_critical_candidate_cluster_ids": critical_ids,
            "capacity_critical_candidate_count": len(critical_ids),
            "semantic_dependence_confirmed_count": 0,
            "required_declared_clusters_present": all(
                declaration["id"]
                in {
                    cluster["cluster_id"]
                    for cluster in declared_results
                }
                for declaration in DECLARED_REVIEW_CLUSTERS
            ),
        },
        "limitations": [
            (
                "Shared words can reflect necessary domain terminology rather "
                "than a shared solution path; lexical scores never confirm "
                "family dependence."
            ),
            (
                "Different words can conceal the same reasoning operation, so "
                "this transparent lexical screen has false negatives."
            ),
            (
                "The exact collapse measures released objective capacity under "
                "a counterfactual assumption; it does not estimate human item "
                "response dependence or teaching efficacy."
            ),
            (
                "Independent semantic review and later learner-response data "
                "remain necessary before changing family declarations."
            ),
        ],
    }
    return {
        **deterministic,
        "artifact_sha256": canonical_hash(deterministic),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--stdout", action="store_true")
    result.add_argument(
        "--fail-on-critical",
        action="store_true",
        help=(
            "Exit 3 when a candidate cluster would reduce exact capacity if "
            "semantic dependence were confirmed."
        ),
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    report = build_report(arguments.corpus)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if arguments.stdout:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "artifact_sha256": report["artifact_sha256"],
                    "findings": report["findings"],
                    "output": str(arguments.output),
                },
                indent=2,
                sort_keys=True,
            )
        )
    if (
        arguments.fail_on_critical
        and report["findings"]["capacity_critical_candidate_count"]
    ):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
