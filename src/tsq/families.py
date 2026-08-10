# SPDX-License-Identifier: MPL-2.0

"""Reviewed equivalence groups for answer-redacted question families.

Question ``family_id`` is part of an item's immutable published identity.  A
later review can nevertheless discover that two published family labels use
the same solution path.  This manifest preserves those historical identities
while assigning one evidence family for selection, dependence discounting,
capacity, and certification.

Every alias member is explicit.  That makes a newly authored item using an old
alias fail closed until its family relationship is reviewed and added here.
"""

from __future__ import annotations

from dataclasses import dataclass
from sqlite3 import Connection


@dataclass(frozen=True, slots=True)
class FamilyAssignment:
    published_family_id: str
    evidence_family_id: str


_ALIASES = {
    "f_transformer_axis_mixing": "f_transformer_removed_attention_audit",
    "f_causal_mask_training_leak": "f_causal_mask_batch_matrix",
    "f_causal_mask_parallelism": "f_causal_mask_batch_matrix",
    "f_causal_mask_matrix": "f_causal_mask_batch_matrix",
    "f_attention_order_label_impossibility": "f_transformer_invariances",
    "f_attention_permutation_contract": "f_transformer_invariances",
    "f_attention_permutation_unshuffle": "f_transformer_invariances",
    "f_attention_permutation_matrix_calculation": (
        "f_transformer_invariances"
    ),
    "f_transformer_position_signal_ablation": "f_transformer_invariances",
    "f_attention_scaled_variance_nonunit": "f_attention_scaling_variance",
    "f_attention_scaling_numeric": "f_attention_scaling_rank",
    "f_attention_double_scaling": "f_attention_scaling_rank",
    "f_attention_window_resource_contrast": "f_transformer_complexity",
    "f_transformer_residual_relu_order": (
        "f_transformer_residual_composition_trace"
    ),
    "f_transformer_residual_parallelization_audit": (
        "f_transformer_residual_composition_trace"
    ),
    "f_attention_value_perturbation": "f_attention_value_role_ablation",
    "f_attention_value_projection_counterfactual": (
        "f_attention_value_role_ablation"
    ),
    "f_ar_fixed_weight_demonstrations": "f_ar_prompt_conditioning",
    "f_agent_tool_data_control_boundary": "f_agent_untrusted_tool_output",
}


_ALIAS_MEMBERS = {
    "q_agent_nested_tool_authority_001": "f_agent_tool_data_control_boundary",
    "q_agent_tool_data_control_boundary_001": "f_agent_tool_data_control_boundary",
    "q_agent_tool_output_argument_injection_001": (
        "f_agent_tool_data_control_boundary"
    ),
    "q_ar_demonstration_checksum_001": "f_ar_fixed_weight_demonstrations",
    "q_ar_demonstration_mapping_swap_001": "f_ar_fixed_weight_demonstrations",
    "q_ar_demonstration_persistence_probe_001": (
        "f_ar_fixed_weight_demonstrations"
    ),
    "q_ar_fixed_weight_demonstrations_001": "f_ar_fixed_weight_demonstrations",
    "q_ar_label_mapping_demonstrations_001": "f_ar_fixed_weight_demonstrations",
    "q_ar_prompt_finetune_control_001": "f_ar_fixed_weight_demonstrations",
    "q_attention_equivariance_contract_002": "f_attention_permutation_contract",
    "q_attention_equivariance_contract_003": "f_attention_permutation_unshuffle",
    "q_attention_equivariance_contract_004": "f_attention_permutation_contract",
    "q_attention_order_label_impossibility_001": (
        "f_attention_order_label_impossibility"
    ),
    "q_attention_order_pooling_002": "f_attention_order_label_impossibility",
    "q_attention_order_pooling_003": "f_attention_order_label_impossibility",
    "q_attention_order_pooling_004": "f_attention_order_label_impossibility",
    "q_attention_output_projection_routing_001": (
        "f_attention_value_projection_counterfactual"
    ),
    "q_attention_output_projection_routing_002": (
        "f_attention_value_projection_counterfactual"
    ),
    "q_attention_output_projection_routing_003": (
        "f_attention_value_projection_counterfactual"
    ),
    "q_attention_permutation_contract_001": "f_attention_permutation_contract",
    "q_attention_permutation_matrix_001": (
        "f_attention_permutation_matrix_calculation"
    ),
    "q_attention_permutation_matrix_002": (
        "f_attention_permutation_matrix_calculation"
    ),
    "q_attention_permutation_matrix_003": (
        "f_attention_permutation_matrix_calculation"
    ),
    "q_attention_permutation_matrix_004": (
        "f_attention_permutation_matrix_calculation"
    ),
    "q_attention_permutation_unshuffle_001": "f_attention_permutation_unshuffle",
    "q_attention_scaled_variance_nonunit_001": (
        "f_attention_scaled_variance_nonunit"
    ),
    "q_attention_scaling_nonunit_variance_001": (
        "f_attention_scaled_variance_nonunit"
    ),
    "q_attention_scaling_numeric_001": "f_attention_scaling_numeric",
    "q_attention_double_scaling_001": "f_attention_double_scaling",
    "q_attention_scaling_linear_divisor_001": "f_attention_double_scaling",
    "q_attention_value_perturbation_001": "f_attention_value_perturbation",
    "q_attention_value_projection_counterfactual_001": (
        "f_attention_value_projection_counterfactual"
    ),
    "q_attention_window_resource_contrast_001": (
        "f_attention_window_resource_contrast"
    ),
    "q_attention_window_resource_contrast_002": (
        "f_attention_window_resource_contrast"
    ),
    "q_attention_window_resource_contrast_003": (
        "f_attention_window_resource_contrast"
    ),
    "q_attention_window_resource_contrast_004": (
        "f_attention_window_resource_contrast"
    ),
    "q_causal_mask_matrix_001": "f_causal_mask_matrix",
    "q_causal_mask_matrix_002": "f_causal_mask_matrix",
    "q_causal_mask_parallelism_001": "f_causal_mask_parallelism",
    "q_causal_mask_training_leak_001": "f_causal_mask_training_leak",
    "q_transformer_axis_mixing_002": "f_transformer_axis_mixing",
    "q_transformer_position_signal_ablation_001": (
        "f_transformer_position_signal_ablation"
    ),
    "q_transformer_position_signal_ablation_002": (
        "f_transformer_position_signal_ablation"
    ),
    "q_transformer_residual_parallelization_audit_001": (
        "f_transformer_residual_parallelization_audit"
    ),
    "q_transformer_residual_relu_order_001": (
        "f_transformer_residual_relu_order"
    ),
    "q_transformer_token_mixing_001": "f_transformer_axis_mixing",
    "q_transformer_token_mixing_002": "f_transformer_axis_mixing",
    "q_transformer_unexpected_cross_token_path_001": (
        "f_transformer_axis_mixing"
    ),
    "q_transformer_unexpected_cross_token_path_002": (
        "f_transformer_axis_mixing"
    ),
}


# Complete answer-redacted cohorts accepted by the 2026-08-09 semantic audit.
# These sets include both the canonical-label questions and every reviewed
# alias member.  Exact membership matters: adding another practice variant to
# one of these already-large families re-enables the quality warning until the
# expanded cohort receives another review.
_REVIEWED_FAMILY_COHORTS = {
    "f_transformer_removed_attention_audit": frozenset({
        "q_transformer_removed_attention_audit_001",
        "q_transformer_axis_mixing_002",
        "q_transformer_axis_mixing_003",
        "q_transformer_axis_mixing_004",
        "q_transformer_attention_ablation_002",
        "q_transformer_attention_ablation_003",
        "q_transformer_attention_ablation_004",
        "q_transformer_token_mixing_002",
        "q_transformer_unexpected_cross_token_path_002",
    }),
    "f_transformer_invariances": frozenset({
        "q_attention_permutation_001",
        "q_attention_order_label_impossibility_001",
        "q_attention_order_pooling_002",
        "q_attention_order_pooling_003",
        "q_attention_order_pooling_004",
        "q_attention_permutation_contract_001",
        "q_attention_permutation_matrix_001",
        "q_attention_permutation_matrix_002",
        "q_attention_permutation_matrix_003",
        "q_attention_permutation_matrix_004",
        "q_attention_equivariance_contract_002",
        "q_attention_equivariance_contract_003",
        "q_attention_equivariance_contract_004",
        "q_attention_permutation_unshuffle_001",
        "q_transformer_position_signal_ablation_002",
    }),
    "f_attention_value_role_ablation": frozenset({
        "q_attention_value_role_ablation_001",
        "q_attention_role_wiring_audit_001",
        "q_attention_role_wiring_audit_002",
        "q_attention_role_wiring_audit_003",
        "q_attention_value_perturbation_001",
        "q_attention_value_projection_counterfactual_001",
        "q_attention_output_projection_routing_001",
        "q_attention_output_projection_routing_002",
        "q_attention_output_projection_routing_003",
    }),
    "f_ar_prompt_conditioning": frozenset({
        "q_ar_prompt_conditioning_001",
        "q_ar_prefix_extension_conditioning_001",
        "q_ar_batched_prompt_conditioning_001",
        "q_ar_context_ablation_fixed_weights_001",
        "q_ar_fixed_weight_demonstrations_001",
        "q_ar_demonstration_checksum_001",
        "q_ar_label_mapping_demonstrations_001",
        "q_ar_prompt_finetune_control_001",
        "q_ar_demonstration_persistence_probe_001",
        "q_ar_demonstration_mapping_swap_001",
    }),
}


def family_assignment(question_id: str, family_id: str) -> FamilyAssignment:
    """Resolve an immutable *published* label to its evidence group.

    A manifested question ID has one exact historical family label.  The
    canonical evidence label is deliberately not accepted here: callers that
    hydrate or publish question-registry rows must prove the immutable label,
    rather than silently repairing a changed row.  Attempt/replay validation
    uses :func:`evidence_family_id`, which accepts either historical or
    canonical evidence labels for manifested IDs.
    """

    published = _ALIAS_MEMBERS.get(question_id)
    if published is not None:
        evidence = _ALIASES[published]
        if family_id != published:
            raise ValueError(
                f"Question {question_id} has family {family_id}, expected "
                f"immutable published family {published}."
            )
        return FamilyAssignment(published, evidence)
    if family_id in _ALIASES:
        raise ValueError(
            f"Question {question_id} uses reviewed family alias {family_id} "
            "without an explicit family-manifest entry."
        )
    return FamilyAssignment(family_id, family_id)


def evidence_family_id(question_id: str, family_id: str) -> str:
    """Resolve historical attempt/replay evidence to its canonical family.

    Attempts created before family equivalence review retain the published
    alias; newer attempts can contain the canonical evidence label.  Both are
    valid evidence encodings for an explicitly manifested question, while a
    question-registry row itself remains subject to ``family_assignment``'s
    exact published-label check.
    """

    published = _ALIAS_MEMBERS.get(question_id)
    if published is None:
        return family_assignment(question_id, family_id).evidence_family_id
    evidence = _ALIASES[published]
    if family_id not in {published, evidence}:
        raise ValueError(
            f"Question {question_id} has evidence family {family_id}, expected "
            f"published family {published} or canonical family {evidence}."
        )
    return evidence


def canonical_family_label(family_id: str) -> str:
    """Resolve a family label when no question identity is available."""

    return _ALIASES.get(family_id, family_id)


def register_family_sql_functions(connection: Connection) -> None:
    """Expose the reviewed family equivalence map to SQLite queries.

    Published rows and immutable learner history retain their original family
    labels.  Runtime queries use this deterministic scalar function to group
    those labels without rewriting historical projections or event hashes.
    """

    connection.create_function(
        "tsq_canonical_family",
        1,
        canonical_family_label,
        deterministic=True,
    )


def published_family_id(question_id: str, family_id: str) -> str:
    return family_assignment(question_id, family_id).published_family_id


def family_alias_members() -> dict[str, FamilyAssignment]:
    """Return a copy for deterministic audits and migrations."""

    return {
        question_id: FamilyAssignment(published, _ALIASES[published])
        for question_id, published in sorted(_ALIAS_MEMBERS.items())
    }


def reviewed_large_family_cohort(
    family_id: str,
    question_ids: set[str],
) -> bool:
    """Return whether a large family exactly matches its reviewed cohort."""

    expected = _REVIEWED_FAMILY_COHORTS.get(family_id)
    return expected is not None and question_ids == expected
