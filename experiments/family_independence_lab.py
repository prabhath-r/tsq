#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0

"""Falsify declared question-family independence with transparent evidence.

The production selector treats distinct evidence-family IDs as independent
opportunities for diagnosis, repair, and verification. This laboratory asks a
narrower, adversarial question: which active cross-family items look similar
enough that a human reviewer should verify that they require genuinely
different solution paths?

Token overlap is deliberately used only to nominate candidates. It is not a
semantic model and cannot establish dependence. For every declared or reviewed
signal cluster, the lab also stress-tests the exact sustained-capacity analyzer
by counterfactually treating the cluster as one family. A capacity drop means
independence is operationally important under that counterfactual; it does not
confirm dependence itself.

Reviewed equivalences are validated against TSQ's explicit family manifest.
The lab never changes question status, invents candidate availability, or
activates content in memory. It also runs the same deterministic corpus-quality
audit used by the release path and exposes every exact warning and error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
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
from tsq.corpus import (  # noqa: E402
    corpus_source_digest,
    load_bundle,
    read_and_parse,
)
from tsq.families import (  # noqa: E402
    family_alias_members,
    family_assignment,
    reviewed_large_family_cohort,
)
from tsq.graph import KnowledgeGraph  # noqa: E402
from tsq.models import LearningObjective, Question  # noqa: E402
from tsq.policy import POLICY_VERSION  # noqa: E402
from tsq.quality import audit_corpus  # noqa: E402
from tsq.versions import DEFAULT_LEARNER_MODEL_VERSION  # noqa: E402


LAB_VERSION = "family-independence-falsification-v6"
REPORT_CONTRACT_VERSION = "family-independence-report-v3"
ANSWER_REDACTED_REVIEW_VERSION = "solution-operation-review-2026-08-10-v1"
NORMALIZATION_VERSION = "lower-alnum-stopwords-v1"
DEFAULT_CORPUS = PROJECT_ROOT / "corpus"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "experiments" / "results" / "family_independence_lab.json"
)
STEM_OVERLAP_THRESHOLD = 0.27
SOLUTION_OVERLAP_THRESHOLD = 0.37
COMBINED_JACCARD_THRESHOLD = 0.29
LARGE_FAMILY_THRESHOLD = 8

# These declarations keep known, high-consequence cases durable even if a
# wording edit pushes a lexical score below the automatic threshold. A reviewed
# equivalence must resolve to one evidence family in the manifest. A reviewed-
# distinct cluster must expose a complete solution-operation partition and
# receives a collapse stress test without changing its adjudication.
DECLARED_REVIEW_CLUSTERS: tuple[dict[str, object], ...] = (
    {
        "id": "attention_value_routing_trio",
        "question_ids": (
            "q_attention_value_role_ablation_001",
            "q_attention_value_projection_counterfactual_001",
            "q_attention_value_perturbation_001",
        ),
        "disposition": "reviewed_equivalent",
        "review_reason": (
            "The semantic review found the three value-vector perturbation "
            "traces to exercise one evidence family. Their immutable published "
            "labels are preserved while runtime evidence is consolidated."
        ),
        "solution_operations": (
            {
                "evidence_family_ids": ("f_attention_value_role_ablation",),
                "operation": (
                    "Hold the query-key routing weights fixed, perturb the "
                    "value path, and propagate the change through the weighted "
                    "value sum."
                ),
            },
        ),
    },
    {
        "id": "multiquery_kv_cache_trio",
        "question_ids": (
            "q_transformer_multiquery_cache_axes_001",
            "q_transformer_multiquery_cache_inventory_001",
            "q_transformer_multiquery_state_transition_001",
        ),
        "disposition": "reviewed_distinct",
        "review_reason": (
            "Answer-redacted review found three different keyed operations: "
            "compare architectural head axes, construct a concrete tensor "
            "inventory, and update persistent versus transient state."
        ),
        "solution_operations": (
            {
                "evidence_family_ids": (
                    "f_transformer_multiquery_cache_axes",
                ),
                "operation": (
                    "Compare multi-head and multi-query architectures along "
                    "the query and key-value head axes while retaining time."
                ),
            },
            {
                "evidence_family_ids": (
                    "f_transformer_multiquery_cache_inventory",
                ),
                "operation": (
                    "Construct the persistent K/V and transient Q tensor "
                    "inventory with exact sequence, head, and width axes."
                ),
            },
            {
                "evidence_family_ids": (
                    "f_transformer_multiquery_state_transition",
                ),
                "operation": (
                    "Append the current shared K/V, discard the used Q, and "
                    "form fresh query heads for the next step."
                ),
            },
        ),
    },
    {
        "id": "causal_training_visibility_triad",
        "question_ids": (
            "q_causal_mask_training_leak_001",
            "q_causal_mask_parallelism_001",
            "q_causal_mask_batch_matrix_001",
        ),
        "disposition": "reviewed_equivalent",
        "review_reason": (
            "The semantic review found that the three prompts exercise the "
            "same inclusive lower-triangular visibility rule. Their immutable "
            "published labels are preserved while evidence is consolidated."
        ),
        "solution_operations": (
            {
                "evidence_family_ids": ("f_causal_mask_batch_matrix",),
                "operation": (
                    "Apply the same inclusive lower-triangular visible-key "
                    "set to each teacher-forced query row."
                ),
            },
        ),
    },
)

# These exact sets were independently adjudicated from stems and option text.
# Correctness flags, rationales, misconception routes, family labels, and
# provenance were excluded from the review view. A reviewed-distinct set must
# retain exactly one contained automatic component spanning every surviving
# evidence family; reviewed-equivalent sets must be fully collapsed by the
# explicit family manifest. Any component that crosses a reviewed set fails
# closed.
REVIEWED_SIGNAL_CLUSTERS: tuple[dict[str, object], ...] = (
    {
        "id": "attention_temperature_vs_variance",
        "question_ids": (
            "q_attention_double_scaling_001",
            "q_attention_scaling_numeric_001",
            "q_attention_scaling_variance_001",
        ),
        "disposition": "reviewed_distinct",
        "review_reason": (
            "The first two prompts apply a common positive logit temperature "
            "and are consolidated; the third derives variance growth under "
            "independent coordinate products."
        ),
        "solution_operations": (
            {
                "evidence_family_ids": ("f_attention_scaling_rank",),
                "operation": (
                    "Divide one realized logit row by a common positive scale, "
                    "compute the new logits, preserve rank, and infer a flatter "
                    "softmax."
                ),
            },
            {
                "evidence_family_ids": ("f_attention_scaling_variance",),
                "operation": (
                    "Derive variance of an independent coordinate-product sum "
                    "and cancel its dimension factor with square-root scaling."
                ),
            },
        ),
    },
    {
        "id": "attention_equivariance_operation_partition",
        "question_ids": (
            "q_attention_equivariance_contract_002",
            "q_attention_equivariance_contract_003",
            "q_attention_permutation_jacobian_001",
            "q_attention_permutation_matrix_002",
            "q_attention_permutation_matrix_004",
            "q_attention_stochastic_permutation_coupling_001",
        ),
        "disposition": "reviewed_distinct",
        "review_reason": (
            "The contract and matrix prompts all apply the same row-"
            "equivariance identity and are consolidated; the Jacobian prompt "
            "differentiates that identity, while the stochastic prompt pushes "
            "an exchangeable mask law through it and constructs a coupling."
        ),
        "solution_operations": (
            {
                "evidence_family_ids": ("f_transformer_invariances",),
                "operation": (
                    "Apply H(PX)=P H(X), equivalently transform attention as "
                    "P A P-transpose and values as P V."
                ),
            },
            {
                "evidence_family_ids": (
                    "f_attention_permutation_jacobian_audit",
                ),
                "operation": (
                    "Differentiate the equivariance identity and conjugate the "
                    "Jacobian across output-token and input-token block axes."
                ),
            },
            {
                "evidence_family_ids": (
                    "f_attention_stochastic_permutation_coupling",
                ),
                "operation": (
                    "Push an iid dropout-mask law through a row permutation "
                    "and couple masks to separate distributional equivariance "
                    "from samplewise equality."
                ),
            },
        ),
    },
    {
        "id": "attention_equivariance_matrix_equivalence",
        "question_ids": (
            "q_attention_permutation_contract_001",
            "q_attention_permutation_matrix_001",
        ),
        "disposition": "reviewed_equivalent",
        "review_reason": (
            "The supplied matrix transformation simplifies exactly to the "
            "same H(PX)=P H(X) assertion as the abstract contract."
        ),
        "solution_operations": (
            {
                "evidence_family_ids": ("f_transformer_invariances",),
                "operation": (
                    "Substitute P A P-transpose and P V, cancel P-transpose P, "
                    "and obtain the permuted output P A V."
                ),
            },
        ),
    },
    {
        "id": "attention_scaling_operation_partition",
        "question_ids": (
            "q_attention_scaled_variance_nonunit_001",
            "q_attention_scaling_covariance_audit_001",
            "q_attention_scaling_covariance_claim_001",
            "q_attention_scaling_entropy_comparison_001",
            "q_attention_scaling_head_dimension_comparison_001",
            "q_attention_scaling_linear_divisor_001",
            "q_attention_scaling_log_odds_001",
            "q_attention_scaling_nonunit_variance_001",
            "q_attention_scaling_rank_warmup_001",
            "q_attention_scaling_softmax_gradient_001",
            "q_attention_scaling_variance_warmup_001",
        ),
        "disposition": "reviewed_distinct",
        "review_reason": (
            "Answer-redacted review separates realized-row temperature effects, "
            "independent-sum variance, covariance corrections, and local "
            "softmax sensitivity. The linear-divisor prompt is consolidated "
            "with the temperature family because its keyed diagnosis is "
            "positive-scale rank preservation plus over-flattening."
        ),
        "solution_operations": (
            {
                "evidence_family_ids": ("f_attention_scaling_rank",),
                "operation": (
                    "Apply a common positive scale to realized logits and "
                    "derive preserved order, log odds, or entropy change."
                ),
            },
            {
                "evidence_family_ids": ("f_attention_scaling_variance",),
                "operation": (
                    "Compute variance of an independent sum and transform it "
                    "by the square of the logit divisor."
                ),
            },
            {
                "evidence_family_ids": (
                    "f_attention_scaling_covariance_audit",
                ),
                "operation": (
                    "Add cross-coordinate covariance terms and reject a unit-"
                    "variance guarantee that assumes independence."
                ),
            },
            {
                "evidence_family_ids": ("f_attention_unscaled_softmax",),
                "operation": (
                    "Evaluate p(1-p) before and after scaling to audit local "
                    "softmax sensitivity under saturation."
                ),
            },
        ),
    },
    {
        "id": "attention_resource_pair_count_equivalence",
        "question_ids": (
            "q_attention_sequence_scaling_001",
            "q_attention_window_resource_contrast_002",
        ),
        "disposition": "reviewed_equivalent",
        "review_reason": (
            "Both prompts solve by multiplying the number of queries by keys "
            "scored per query while holding projection parameters fixed."
        ),
        "solution_operations": (
            {
                "evidence_family_ids": ("f_transformer_complexity",),
                "operation": (
                    "Count L times L pairs for full attention or L times a "
                    "bounded window, independently of parameter count."
                ),
            },
        ),
    },
    {
        "id": "causal_visibility_operation_partition",
        "question_ids": (
            "q_causal_full_incremental_intermediate_row_001",
            "q_causal_mask_first_row_001",
            "q_causal_packed_visible_set_001",
        ),
        "disposition": "reviewed_distinct",
        "review_reason": (
            "The prompts respectively compare two execution-mode key sets, "
            "normalize a masked logit row, and intersect a causal prefix with "
            "segment membership."
        ),
        "solution_operations": (
            {
                "evidence_family_ids": (
                    "f_causal_full_incremental_equivalence",
                ),
                "operation": (
                    "Compare full-pass and incremental visible K/V sets and "
                    "infer equal row outputs when the sets and parameters match."
                ),
            },
            {
                "evidence_family_ids": (
                    "f_causal_mask_softmax_normalization",
                ),
                "operation": (
                    "Mask future logits before softmax and normalize only over "
                    "the inclusive visible prefix."
                ),
            },
            {
                "evidence_family_ids": ("f_causal_mask_packed_sequences",),
                "operation": (
                    "Intersect inclusive causal visibility with same-segment "
                    "membership for a packed query."
                ),
            },
        ),
    },
    {
        "id": "kv_cache_operation_partition",
        "question_ids": (
            "q_kv_cache_cross_attention_static_memory_001",
            "q_kv_cache_prompt_transition_warmup_001",
            "q_kv_cache_query_head_intervention_001",
            "q_kv_cache_scalar_inventory_001",
            "q_transformer_kv_cache_alignment_002",
            "q_transformer_kv_cache_eviction_equivalence_002",
            "q_transformer_multiquery_cache_axes_001",
            "q_transformer_multiquery_cache_inventory_001",
            "q_transformer_multiquery_state_transition_001",
        ),
        "disposition": "reviewed_distinct",
        "review_reason": (
            "The five evidence families require architectural axis comparison, "
            "tensor inventory, lifetime transition, K/V positional alignment, "
            "or an equivalence judgment after context eviction."
        ),
        "solution_operations": (
            {
                "evidence_family_ids": (
                    "f_transformer_multiquery_cache_axes",
                ),
                "operation": (
                    "Compare query-head and shared K/V-head axes while "
                    "preserving the sequence axis."
                ),
            },
            {
                "evidence_family_ids": (
                    "f_transformer_multiquery_cache_inventory",
                ),
                "operation": (
                    "Write persistent and transient tensor shapes or multiply "
                    "their axes to obtain scalar inventory."
                ),
            },
            {
                "evidence_family_ids": (
                    "f_transformer_multiquery_state_transition",
                ),
                "operation": (
                    "Distinguish reusable K/V from fresh Q and update or reuse "
                    "state across decoding steps."
                ),
            },
            {
                "evidence_family_ids": ("f_incremental_cache_kv_alignment",),
                "operation": (
                    "Verify that each attention coefficient and value row use "
                    "the same temporal index ordering."
                ),
            },
            {
                "evidence_family_ids": ("f_kv_cache_eviction_equivalence",),
                "operation": (
                    "Compare visible contexts and reject exact logit or future-"
                    "trajectory equivalence after eviction from an otherwise "
                    "full-prefix computation."
                ),
            },
        ),
    },
    {
        "id": "normalization_residual_operation_partition",
        "question_ids": (
            "q_transformer_norm_residual_placement_001",
            "q_transformer_pre_post_ln_forms_001",
        ),
        "disposition": "reviewed_distinct",
        "review_reason": (
            "One prompt audits a non-equivalent residual/normalization refactor; "
            "the other identifies the two canonical Pre-LN and Post-LN "
            "architectural equations."
        ),
        "solution_operations": (
            {
                "evidence_family_ids": (
                    "f_transformer_norm_residual_placement",
                ),
                "operation": (
                    "Compare LayerNorm(x+S(x)) with x+LayerNorm(S(x)) and reject "
                    "commutation of normalization with residual addition."
                ),
            },
            {
                "evidence_family_ids": (
                    "f_transformer_pre_post_ln_contract",
                ),
                "operation": (
                    "Identify Post-LN as LayerNorm(x+S(x)) and Pre-LN as "
                    "x+S(LayerNorm(x))."
                ),
            },
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
_SERVICEABILITY_CODES = frozenset(
    {
        "missing_primary_mapping_coverage",
        "insufficient_primary_family_coverage",
        "insufficient_contextual_family_coverage",
        "insufficient_objective_family_coverage",
        "unserviceable_objective_path",
    }
)
_ISSUE_SUBJECT_PATTERNS = (
    re.compile(r"^Learning objective ([a-z0-9_]+)\b"),
    re.compile(r"^Root ([a-z0-9_]+)\b"),
    re.compile(r"^Family ([a-z0-9_]+)\b"),
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


def _published_family_id(question: Question) -> str:
    return question.published_family_id or question.family_id


def pair_evidence(
    left: QuestionFeatures, right: QuestionFeatures
) -> dict[str, object]:
    if left.question.id == right.question.id:
        raise ValueError("Pair evidence requires two different questions.")
    if left.question.id > right.question.id:
        left, right = right, left
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
    different_evidence_families = (
        left.question.family_id != right.question.family_id
    )
    lexical_threshold_passed = bool(
        (
            stem["overlap_coefficient"] >= STEM_OVERLAP_THRESHOLD
            and solution["overlap_coefficient"]
            >= SOLUTION_OVERLAP_THRESHOLD
        )
        or combined["jaccard"] >= COMBINED_JACCARD_THRESHOLD
    )
    qualifies = bool(
        same_objective
        and same_misconceptions
        and different_evidence_families
        and lexical_threshold_passed
    )
    return {
        "left_question_id": left.question.id,
        "right_question_id": right.question.id,
        "left_published_family_id": _published_family_id(left.question),
        "right_published_family_id": _published_family_id(right.question),
        "left_family_id": left.question.family_id,
        "right_family_id": right.question.family_id,
        "different_evidence_families": different_evidence_families,
        "objective_id": left.question.objective_id if same_objective else None,
        "identical_named_misconception_set": same_misconceptions,
        "named_misconception_ids": (
            list(left.misconception_ids) if same_misconceptions else []
        ),
        "stem_overlap": stem,
        "solution_path_overlap": solution,
        "combined_overlap": combined,
        "lexical_threshold_passed": lexical_threshold_passed,
        "signal_threshold_passed": qualifies,
        "candidate_status": (
            "requires_independent_semantic_review"
            if qualifies
            else (
                "already_one_reviewed_evidence_family"
                if not different_evidence_families
                else "below_automatic_nomination_threshold"
            )
        ),
        "semantic_dependence_established_by_lexical_signal": False,
        # Backward-compatible truth value: lexical evidence itself never proves
        # dependence, including when an external semantic review already did.
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
    return analyze_sustained_capacity(
        objective_questions,
        graph,
        misconceptions,
        (target,),
        unavailable_family_ids=unavailable_family_ids,
    ).targets[0]


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
    signatures = {
        tuple(sorted(question.misconception_ids)) for question in selected
    }
    if len(objective_ids) != 1 or None in objective_ids or len(signatures) != 1:
        raise LabInvariantError(
            "A counterfactual cluster must share one fine objective and one "
            "named-misconception signature."
        )
    objective_id = next(iter(objective_ids))
    misconception_ids = next(iter(signatures))
    if not misconception_ids:
        raise LabInvariantError(
            "A counterfactual cluster needs a named misconception signature."
        )
    objective = objectives[objective_id]
    family_ids = tuple(sorted({question.family_id for question in selected}))
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

    if len(family_ids) == 1:
        snapshot = _capacity_snapshot(baseline)
        return {
            "scope": (
                "active_questions_with_same_fine_objective_and_identical_"
                "named_misconception_set"
            ),
            "objective_id": objective_id,
            "objective_name": objective.name,
            "counterfactual_evaluated": False,
            "counterfactual_assumption": None,
            "retained_representative_family_id": family_ids[0],
            "counterfactually_unavailable_family_ids": [],
            "baseline": snapshot,
            "collapsed": snapshot,
            "impact": {
                "initial_safe_family_drop": 0,
                "order_robust_main_capacity_drop": 0,
                "achievable_main_capacity_drop": 0,
                "capacity_critical_if_dependence_confirmed": False,
                "assessment": "already_collapsed_by_semantic_review",
            },
        }

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
        "counterfactual_evaluated": True,
        "counterfactual_assumption": (
            "Treat the nominated evidence families as one semantic family by "
            "retaining one deterministic representative and removing the rest."
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
    rows: list[dict[str, object]] = []
    for index, left_id in enumerate(question_ids):
        for right_id in question_ids[index + 1 :]:
            rows.append(pair_evidence(features[left_id], features[right_id]))
    return sorted(
        rows,
        key=lambda row: (row["left_question_id"], row["right_question_id"]),
    )


def _review_fields(
    declaration: dict[str, object],
    evidence_family_ids: Sequence[str],
) -> dict[str, object]:
    """Validate and expose one exact answer-redacted adjudication."""

    disposition = str(declaration["disposition"])
    if disposition not in {"reviewed_distinct", "reviewed_equivalent"}:
        raise LabInvariantError(
            f"Review cluster {declaration['id']} has unsupported disposition "
            f"{disposition}."
        )
    operations = tuple(declaration.get("solution_operations", ()))
    if not operations:
        raise LabInvariantError(
            f"Review cluster {declaration['id']} lacks solution operations."
        )
    operation_family_ids: list[str] = []
    normalized_operations: list[dict[str, object]] = []
    for value in operations:
        if not isinstance(value, dict):
            raise LabInvariantError(
                f"Review cluster {declaration['id']} has a malformed solution "
                "operation."
            )
        family_ids = tuple(str(item) for item in value["evidence_family_ids"])
        operation = str(value["operation"]).strip()
        if not family_ids or not operation:
            raise LabInvariantError(
                f"Review cluster {declaration['id']} has an incomplete solution "
                "operation."
            )
        operation_family_ids.extend(family_ids)
        normalized_operations.append(
            {
                "evidence_family_ids": list(family_ids),
                "operation": operation,
            }
        )
    expected_families = sorted(set(evidence_family_ids))
    if (
        len(operation_family_ids) != len(set(operation_family_ids))
        or sorted(operation_family_ids) != expected_families
    ):
        raise LabInvariantError(
            f"Review cluster {declaration['id']} solution operations do not "
            "partition its exact evidence families."
        )

    reviewed_equivalent = disposition == "reviewed_equivalent"
    if reviewed_equivalent != (len(expected_families) == 1):
        raise LabInvariantError(
            f"Review cluster {declaration['id']} disposition does not match "
            "the current family manifest."
        )
    return {
        "review_version": ANSWER_REDACTED_REVIEW_VERSION,
        "review_input": "stem_and_option_text_only",
        "review_reason": str(declaration["review_reason"]),
        "review_status": (
            "reviewed_equivalent_in_family_manifest"
            if reviewed_equivalent
            else "reviewed_distinct_after_answer_redacted_review"
        ),
        "solution_operations": normalized_operations,
        "semantic_dependence_established": reviewed_equivalent,
        "semantic_independence_established": not reviewed_equivalent,
        "semantic_dependence_evidence": (
            "explicit_reviewed_family_manifest"
            if reviewed_equivalent
            else None
        ),
        "semantic_independence_evidence": (
            None
            if reviewed_equivalent
            else "answer_redacted_solution_operation_partition"
        ),
    }


def _family_manifest_audit(
    *,
    questions: Sequence[Question],
    active_questions: Sequence[Question],
) -> dict[str, object]:
    """Validate every explicit alias and every currently large cohort exactly."""

    manifest = family_alias_members()
    by_question_id = {question.id: question for question in questions}
    alias_labels = {
        assignment.published_family_id for assignment in manifest.values()
    }
    violations: list[str] = []
    rows: list[dict[str, object]] = []

    missing = sorted(set(manifest) - set(by_question_id))
    if missing:
        violations.append("missing_alias_members:" + ",".join(missing))

    unlisted_alias_users = sorted(
        question.id
        for question in questions
        if _published_family_id(question) in alias_labels
        and question.id not in manifest
    )
    if unlisted_alias_users:
        violations.append(
            "unlisted_alias_members:" + ",".join(unlisted_alias_users)
        )

    for question_id, expected in manifest.items():
        question = by_question_id.get(question_id)
        if question is None:
            continue
        assignment = family_assignment(
            question.id,
            _published_family_id(question),
        )
        valid = (
            assignment == expected
            and _published_family_id(question) == expected.published_family_id
            and question.family_id == expected.evidence_family_id
        )
        if not valid:
            violations.append(f"assignment_mismatch:{question_id}")
        rows.append(
            {
                "question_id": question_id,
                "status": question.status.value,
                "published_family_id": _published_family_id(question),
                "evidence_family_id": question.family_id,
                "manifest_published_family_id": expected.published_family_id,
                "manifest_evidence_family_id": expected.evidence_family_id,
                "exact_assignment_valid": valid,
            }
        )

    active_members: dict[str, set[str]] = {}
    for question in active_questions:
        active_members.setdefault(question.family_id, set()).add(question.id)
    large_rows: list[dict[str, object]] = []
    for family_id, question_ids in sorted(active_members.items()):
        if len(question_ids) <= LARGE_FAMILY_THRESHOLD:
            continue
        exact = reviewed_large_family_cohort(family_id, question_ids)
        if not exact:
            violations.append(f"unreviewed_large_cohort:{family_id}")
        large_rows.append(
            {
                "evidence_family_id": family_id,
                "question_count": len(question_ids),
                "question_ids": sorted(question_ids),
                "exact_reviewed_cohort": exact,
            }
        )

    if violations:
        raise LabInvariantError(
            "Family manifest validation failed: " + "; ".join(violations)
        )
    return {
        "manifest_semantics": (
            "immutable published family labels map to reviewed evidence "
            "families; every alias-bearing question is explicit"
        ),
        "explicit_alias_member_count": len(manifest),
        "explicit_alias_family_count": len(alias_labels),
        "member_assignments": sorted(rows, key=lambda row: row["question_id"]),
        "large_family_threshold": LARGE_FAMILY_THRESHOLD,
        "reviewed_large_cohorts": large_rows,
        "reviewed_large_cohort_count": len(large_rows),
        "all_alias_assignments_exact": True,
        "all_alias_users_explicit": True,
        "all_large_cohorts_exactly_reviewed": True,
        "fail_closed": True,
    }


def _serviceability_audit(
    *,
    concepts: Sequence[object],
    questions: Sequence[Question],
    graph: KnowledgeGraph,
    misconceptions: Sequence[object],
) -> dict[str, object]:
    issues = audit_corpus(
        questions,
        expected_primary_concept_ids={
            mapping.concept_id
            for question in questions
            for mapping in question.concepts
        },
        knowledge_graph=graph,
        misconceptions=misconceptions,
    )
    rows = [asdict(issue) for issue in issues]

    def identifier(row: dict[str, object]) -> str:
        question_id = row.get("question_id")
        if question_id:
            subject = str(question_id)
        else:
            message = str(row["message"])
            match = next(
                (
                    candidate.match(message)
                    for candidate in _ISSUE_SUBJECT_PATTERNS
                    if candidate.match(message) is not None
                ),
                None,
            )
            subject = (
                match.group(1)
                if match is not None
                else canonical_hash(row)[:12]
            )
        return f"{row['code']}:{subject}"

    for row in rows:
        row["issue_id"] = identifier(row)
    errors = [row for row in rows if row["severity"] == "error"]
    warnings = [row for row in rows if row["severity"] == "warning"]
    serviceability = [
        row for row in rows if row["code"] in _SERVICEABILITY_CODES
    ]

    return {
        "audit_scope": (
            "all parsed questions with active-bank coverage, contextual, and "
            "fine-objective serviceability checks"
        ),
        "concept_count": len(concepts),
        "question_count": len(questions),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "strict_pass": not errors and not warnings,
        "error_codes": sorted({str(row["code"]) for row in errors}),
        "warning_codes": sorted({str(row["code"]) for row in warnings}),
        "error_identifiers": sorted(str(row["issue_id"]) for row in errors),
        "warning_identifiers": sorted(
            str(row["issue_id"]) for row in warnings
        ),
        "serviceability_issue_count": len(serviceability),
        "serviceability_issue_codes": sorted(
            {str(row["code"]) for row in serviceability}
        ),
        "serviceability_issue_identifiers": sorted(
            str(row["issue_id"]) for row in serviceability
        ),
        "errors": errors,
        "warnings": warnings,
        "serviceability_issues": serviceability,
    }


def build_report(corpus: Path = DEFAULT_CORPUS) -> dict[str, object]:
    source_digest_before = corpus_source_digest(corpus)
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
        (row for row in all_pairs if row["signal_threshold_passed"]),
        key=lambda row: (
            row["left_question_id"],
            row["right_question_id"],
        ),
    )
    signal_components = _connected_components(features, qualifying_edges)

    baseline_cache: dict[str, TargetCapacity] = {}
    declared_results: list[dict[str, object]] = []
    for declaration in DECLARED_REVIEW_CLUSTERS:
        question_ids = tuple(str(value) for value in declaration["question_ids"])
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
        evidence_family_ids = sorted(
            {question.family_id for question in selected}
        )
        review_fields = _review_fields(declaration, evidence_family_ids)
        pair_rows = _cluster_pair_evidence(question_ids, features)
        declared_results.append(
            {
                "cluster_id": declaration["id"],
                "origin": "declared_review_seed",
                "question_ids": list(question_ids),
                "published_family_ids": sorted(
                    _published_family_id(question) for question in selected
                ),
                "family_ids": evidence_family_ids,
                "objective_id": selected[0].objective_id,
                "named_misconception_ids": sorted(
                    selected[0].misconception_ids
                ),
                **review_fields,
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
    signal_component_sets = {
        frozenset(question_ids): question_ids
        for question_ids in signal_components
    }
    reviewed_signal_results: list[dict[str, object]] = []
    reviewed_signal_sets: set[frozenset[str]] = set()
    adjudicated_signal_component_sets: set[frozenset[str]] = set()
    for declaration in REVIEWED_SIGNAL_CLUSTERS:
        question_ids = tuple(str(value) for value in declaration["question_ids"])
        question_set = frozenset(question_ids)
        if question_set in reviewed_signal_sets:
            raise LabInvariantError(
                f"Duplicate reviewed signal cluster set at {declaration['id']}."
            )
        reviewed_signal_sets.add(question_set)
        missing = sorted(set(question_ids) - set(by_question_id))
        if missing:
            raise LabInvariantError(
                f"Reviewed signal cluster {declaration['id']} has inactive or "
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
                f"Reviewed signal cluster {declaration['id']} no longer "
                "shares one objective and one non-empty named-misconception "
                "set."
            )
        evidence_family_ids = sorted(
            {question.family_id for question in selected}
        )
        review_fields = _review_fields(declaration, evidence_family_ids)
        pair_rows = _cluster_pair_evidence(question_ids, features)
        automatic_signal_connected = question_set in signal_component_sets
        intersecting_components = [
            component_set
            for component_set in signal_component_sets
            if component_set & question_set
        ]
        if any(
            not component_set.issubset(question_set)
            for component_set in intersecting_components
        ):
            raise LabInvariantError(
                f"Automatic signal membership crossed reviewed cluster "
                f"{declaration['id']}."
            )
        if review_fields["semantic_independence_established"]:
            if len(intersecting_components) != 1:
                raise LabInvariantError(
                    f"Reviewed-distinct signal cluster {declaration['id']} "
                    "does not retain exactly one contained automatic "
                    "nomination."
                )
            component_family_ids = {
                by_question_id[question_id].family_id
                for question_id in next(iter(intersecting_components))
            }
            if component_family_ids != set(evidence_family_ids):
                raise LabInvariantError(
                    f"Reviewed-distinct signal cluster {declaration['id']} "
                    "no longer nominates every reviewed evidence family."
                )
            adjudicated_signal_component_sets.update(intersecting_components)
        elif intersecting_components:
            raise LabInvariantError(
                f"Reviewed-equivalent signal cluster {declaration['id']} was "
                "not collapsed by the family manifest."
            )
        if not any(row["lexical_threshold_passed"] for row in pair_rows):
            raise LabInvariantError(
                f"Reviewed signal cluster {declaration['id']} no longer has "
                "the lexical evidence that originally nominated it."
            )
        reviewed_signal_results.append(
            {
                "cluster_id": declaration["id"],
                "origin": "answer_redacted_signal_adjudication",
                "question_ids": list(question_ids),
                "published_family_ids": sorted(
                    _published_family_id(question) for question in selected
                ),
                "family_ids": evidence_family_ids,
                "objective_id": selected[0].objective_id,
                "named_misconception_ids": sorted(
                    selected[0].misconception_ids
                ),
                **review_fields,
                "automatic_signal_connected": automatic_signal_connected,
                "automatic_signal_component_question_ids": [
                    sorted(component_set)
                    for component_set in intersecting_components
                ],
                "automatic_lexical_pair_present": True,
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

    unresolved_results: list[dict[str, object]] = []
    for question_ids in signal_components:
        question_set = frozenset(question_ids)
        if (
            question_set in declared_sets
            or question_set in adjudicated_signal_component_sets
        ):
            continue
        selected = [by_question_id[question_id] for question_id in question_ids]
        if len({question.objective_id for question in selected}) != 1:
            raise LabInvariantError("Signal component crossed objectives.")
        unresolved_results.append(
            {
                "cluster_id": (
                    f"unresolved_signal_cluster_{len(unresolved_results) + 1:03d}"
                ),
                "origin": "transparent_lexical_signal",
                "question_ids": list(question_ids),
                "published_family_ids": sorted(
                    _published_family_id(question) for question in selected
                ),
                "family_ids": sorted(
                    {question.family_id for question in selected}
                ),
                "objective_id": selected[0].objective_id,
                "named_misconception_ids": sorted(
                    selected[0].misconception_ids
                ),
                "review_status": "independent_semantic_review_required",
                "semantic_dependence_established": False,
                "semantic_independence_established": False,
                "semantic_dependence_evidence": None,
                "semantic_independence_evidence": None,
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

    clusters = [
        *declared_results,
        *reviewed_signal_results,
        *unresolved_results,
    ]
    critical_ids = [
        str(cluster["cluster_id"])
        for cluster in unresolved_results
        if cluster["capacity_stress_test"]["impact"][
            "capacity_critical_if_dependence_confirmed"
        ]
    ]
    reviewed_distinct_critical_ids = [
        str(cluster["cluster_id"])
        for cluster in clusters
        if cluster["semantic_independence_established"]
        and cluster["capacity_stress_test"]["impact"][
            "capacity_critical_if_dependence_confirmed"
        ]
    ]
    family_manifest = _family_manifest_audit(
        questions=questions,
        active_questions=active,
    )
    serviceability = _serviceability_audit(
        concepts=concepts,
        questions=questions,
        graph=graph,
        misconceptions=misconceptions,
    )
    source_digest_after = corpus_source_digest(corpus)
    if source_digest_after != source_digest_before:
        raise LabInvariantError(
            "The source corpus changed while the laboratory was running."
        )

    status_counts = Counter(question.status.value for question in questions)
    deterministic: dict[str, object] = {
        "lab_version": LAB_VERSION,
        "report_contract_version": REPORT_CONTRACT_VERSION,
        "corpus": {
            "path": (
                str(corpus.resolve().relative_to(PROJECT_ROOT))
                if corpus.resolve().is_relative_to(PROJECT_ROOT)
                else str(corpus)
            ),
            "sha256": source_digest_before,
            "source_bytes_unchanged": True,
            "schema_version": raw["schema_version"],
            "title": raw["title"],
            "question_count": len(questions),
            "status_counts": dict(sorted(status_counts.items())),
            "active_question_count": len(active),
            "published_active_family_count": len(
                {_published_family_id(question) for question in active}
            ),
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
                "misconception set, different reviewed evidence families, "
                "and the documented lexical threshold"
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
            "answer_redacted_review": {
                "version": ANSWER_REDACTED_REVIEW_VERSION,
                "included": "question stem and option text",
                "excluded": (
                    "correctness flags, rationales, misconception routes, "
                    "published and evidence family labels, and provenance"
                ),
                "distinct_semantics": (
                    "each retained evidence family has a separately stated "
                    "keyed solution operation"
                ),
                "equivalent_semantics": (
                    "the exact question set is consolidated by the explicit "
                    "family manifest"
                ),
                "membership_drift": "fail_closed",
            },
            "lexical_signal_semantics": (
                "candidate nomination only; lexical overlap does not prove "
                "semantic dependence or interchangeability"
            ),
            "counterfactual_semantics": (
                "capacity consequence under hypothetical dependence; reviewed-"
                "equivalent clusters are already grouped, reviewed-distinct "
                "clusters retain the stress result, and no collapse mutates "
                "the corpus"
            ),
            "family_manifest_semantics": (
                "published labels remain immutable while explicit reviewed "
                "aliases define the evidence families used here"
            ),
            "content_activation_semantics": (
                "none; the lab analyzes only currently adaptation-eligible "
                "questions and never substitutes a status"
            ),
        },
        "duplicate_pair_candidates": duplicate_candidates,
        "declared_review_clusters": declared_results,
        "signal_nominated_clusters": [
            *reviewed_signal_results,
            *unresolved_results,
        ],
        "reviewed_signal_clusters": reviewed_signal_results,
        "unresolved_signal_clusters": unresolved_results,
        "family_manifest_audit": family_manifest,
        "serviceability_audit": serviceability,
        "findings": {
            "eligible_signature_pair_count": len(all_pairs),
            "duplicate_pair_candidate_count": len(duplicate_candidates),
            "declared_review_cluster_count": len(declared_results),
            "reviewed_equivalent_declared_cluster_ids": sorted(
                str(cluster["cluster_id"])
                for cluster in declared_results
                if cluster["semantic_dependence_established"]
            ),
            "reviewed_distinct_declared_cluster_ids": sorted(
                str(cluster["cluster_id"])
                for cluster in declared_results
                if cluster["semantic_independence_established"]
            ),
            "automatic_signal_component_count": len(signal_components),
            "signal_nominated_cluster_count": (
                len(reviewed_signal_results) + len(unresolved_results)
            ),
            "reviewed_signal_cluster_count": len(reviewed_signal_results),
            "reviewed_distinct_signal_cluster_ids": sorted(
                str(cluster["cluster_id"])
                for cluster in reviewed_signal_results
                if cluster["semantic_independence_established"]
            ),
            "reviewed_equivalent_signal_cluster_ids": sorted(
                str(cluster["cluster_id"])
                for cluster in reviewed_signal_results
                if cluster["semantic_dependence_established"]
            ),
            "unresolved_signal_cluster_ids": sorted(
                str(cluster["cluster_id"])
                for cluster in unresolved_results
            ),
            "unresolved_signal_cluster_count": len(unresolved_results),
            "capacity_critical_candidate_cluster_ids": critical_ids,
            "capacity_critical_candidate_count": len(critical_ids),
            "capacity_critical_reviewed_distinct_cluster_ids": sorted(
                reviewed_distinct_critical_ids
            ),
            "semantic_dependence_confirmed_count": sum(
                bool(cluster["semantic_dependence_established"])
                for cluster in clusters
            ),
            "semantic_independence_confirmed_count": sum(
                bool(cluster["semantic_independence_established"])
                for cluster in clusters
            ),
            "explicit_alias_member_count": family_manifest[
                "explicit_alias_member_count"
            ],
            "reviewed_large_cohort_count": family_manifest[
                "reviewed_large_cohort_count"
            ],
            "serviceability_error_count": serviceability["error_count"],
            "serviceability_warning_count": serviceability["warning_count"],
            "serviceability_warning_identifiers": serviceability[
                "warning_identifiers"
            ],
            "strict_corpus_audit_passed": serviceability["strict_pass"],
            "required_declared_clusters_present": all(
                declaration["id"]
                in {
                    cluster["cluster_id"] for cluster in declared_results
                }
                for declaration in DECLARED_REVIEW_CLUSTERS
            ),
            "required_signal_reviews_present": all(
                declaration["id"]
                in {
                    cluster["cluster_id"]
                    for cluster in reviewed_signal_results
                }
                for declaration in REVIEWED_SIGNAL_CLUSTERS
            ),
            "in_memory_status_substitution_count": 0,
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
                "New or membership-changed lexical nominations require another "
                "answer-redacted semantic review before their family "
                "declarations can change."
            ),
            (
                "The operation review adjudicates authored solution paths; "
                "later learner-response data remains necessary to estimate "
                "empirical response dependence."
            ),
            (
                "Exact manifest agreement proves that the reviewed grouping is "
                "applied consistently; it does not independently repeat the "
                "original semantic review."
            ),
            (
                "A strict deterministic corpus audit checks structural clues "
                "and serviceable routing paths, not human calibration or learning "
                "efficacy."
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
            "Exit 3 when an unresolved candidate cluster would reduce exact "
            "capacity if semantic dependence were confirmed."
        ),
    )
    result.add_argument(
        "--fail-on-serviceability",
        action="store_true",
        help="Exit 4 when the deterministic corpus audit has any issue.",
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
    if (
        arguments.fail_on_serviceability
        and not report["findings"]["strict_corpus_audit_passed"]
    ):
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
