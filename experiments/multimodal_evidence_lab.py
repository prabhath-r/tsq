#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0

"""Exercise TSQ's pure multimodal evidence boundary adversarially.

This laboratory is intentionally separate from the unit-test suite and from
the production database.  It feeds deterministic *recorded fixtures* into the
evidence contracts; it does not execute learner code, invoke a model, infer a
score from telemetry, or update a learner projection.  The resulting artifact
is therefore useful for inspecting the exact boundary between observations and
claims before a sandboxed runner or reviewed performance-task registry exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tsq.evidence import (  # noqa: E402
    ActionKind,
    ActionPhase,
    CriterionEvaluation,
    CriterionScale,
    EvaluationStatus,
    EvidenceBundle,
    LearningAction,
    LearningTask,
    RubricCriterion,
    ScorerContract,
    ScorerKind,
    TaskEvaluation,
    TaskModality,
    action_trace_digest,
    canonical_digest,
    canonical_json,
    reduce_evidence,
    summarize_actions,
)


LAB_VERSION = "multimodal-evidence-lab-v1"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "experiments" / "results" / "multimodal_evidence_lab.json"
)


class LabInvariantError(RuntimeError):
    """Raised when an evidence-safety property is contradicted."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LabInvariantError(message)


def digest(label: str) -> str:
    return hashlib.sha256(f"{LAB_VERSION}|{label}".encode("utf-8")).hexdigest()


def action(
    attempt_id: str,
    sequence: int,
    kind: ActionKind,
    payload: Mapping[str, Any],
    *,
    phase: ActionPhase = ActionPhase.UNASSISTED,
    elapsed_ms: int | None = None,
) -> LearningAction:
    suffix = attempt_id.removeprefix("att_")
    return LearningAction(
        id=f"act_{suffix}_{sequence:02d}",
        trace_id=attempt_id,
        sequence=sequence,
        kind=kind,
        phase=phase,
        payload=payload,
        elapsed_ms=elapsed_ms,
    )


def resequence(actions: Iterable[LearningAction]) -> tuple[LearningAction, ...]:
    """Return the same semantic checkpoints with a contiguous counterfactual order."""

    return tuple(
        LearningAction(
            id=item.id,
            trace_id=item.trace_id,
            sequence=index,
            kind=item.kind,
            phase=item.phase,
            payload=item.payload,
            elapsed_ms=item.elapsed_ms,
            schema_version=item.schema_version,
        )
        for index, item in enumerate(actions)
    )


def checkpoint(
    attempt_id: str,
    sequence: int,
    artifact_kind: str,
    *,
    phase: ActionPhase = ActionPhase.UNASSISTED,
    elapsed_ms: int | None = None,
) -> LearningAction:
    return action(
        attempt_id,
        sequence,
        ActionKind.ARTIFACT_CHECKPOINT,
        {
            "artifact_digest": digest(f"{attempt_id}|artifact|{sequence}"),
            "artifact_kind": artifact_kind,
        },
        phase=phase,
        elapsed_ms=elapsed_ms,
    )


def check_run(
    attempt_id: str,
    sequence: int,
    *,
    passed: int,
    failed: int,
    errored: int = 0,
    skipped: int = 0,
    phase: ActionPhase = ActionPhase.UNASSISTED,
    elapsed_ms: int | None = None,
) -> LearningAction:
    return action(
        attempt_id,
        sequence,
        ActionKind.CHECK_RUN,
        {
            "check_set_id": "fixture_checks_v1",
            "passed": passed,
            "failed": failed,
            "errored": errored,
            "skipped": skipped,
            "result_digest": digest(f"{attempt_id}|checks|{sequence}"),
        },
        phase=phase,
        elapsed_ms=elapsed_ms,
    )


def criterion_evaluation(
    criterion_id: str,
    score: float,
    source_action_ids: Sequence[str],
    *,
    phase: ActionPhase = ActionPhase.UNASSISTED,
    scorer_kind: ScorerKind = ScorerKind.DETERMINISTIC,
    scorer_id: str = "reviewed_fixture",
    outcome_code: str = "satisfied",
    reliability: float = 1.0,
) -> CriterionEvaluation:
    return CriterionEvaluation(
        criterion_id=criterion_id,
        status=EvaluationStatus.VALID,
        score=score,
        outcome_code=outcome_code,
        phase=phase,
        scorer_kind=scorer_kind,
        scorer_id=scorer_id,
        scorer_version="v1",
        source_action_ids=tuple(source_action_ids),
        reliability=reliability,
    )


def task_evaluation(
    task: LearningTask,
    attempt_id: str,
    criteria: Iterable[CriterionEvaluation],
    actions: Sequence[LearningAction],
) -> TaskEvaluation:
    return TaskEvaluation(
        id=f"eval_{attempt_id.removeprefix('att_')}",
        trace_id=attempt_id,
        task_id=task.id,
        task_version=task.version,
        task_digest=task.digest,
        action_trace_digest=action_trace_digest(actions),
        criteria=tuple(criteria),
    )


def fixture_scorer_contract(*criterion_ids: str) -> ScorerContract:
    return ScorerContract(
        kind=ScorerKind.DETERMINISTIC,
        scorer_id="reviewed_fixture",
        scorer_version="v1",
        authority_id="authority_reviewed_fixture",
        authority_manifest_digest=digest("authority|reviewed_fixture|v1"),
        criterion_ids=tuple(criterion_ids),
        evidence_action_kinds=(
            ActionKind.ANSWER_REVISED,
            ActionKind.ARTIFACT_CHECKPOINT,
            ActionKind.EXPLANATION_CHECKPOINT,
            ActionKind.CHECK_RUN,
            ActionKind.SUBMITTED,
        ),
        check_set_manifests=(
            ("fixture_checks_v1", digest("check_manifest|fixture_checks_v1")),
        ),
        artifact_manifests=(
            (
                "architecture_diagram",
                digest("artifact_manifest|architecture_diagram"),
            ),
            ("design_snapshot", digest("artifact_manifest|design_snapshot")),
            ("patch_digest", digest("artifact_manifest|patch_digest")),
            ("source_tree", digest("artifact_manifest|source_tree")),
        ),
    )


def task_identity(task_id: str, title: str) -> dict[str, Any]:
    """Return the content-addressed administration terms for a lab fixture."""

    return {
        "instructions": (
            f"{title}. Follow the pinned stimulus and submit only the requested "
            "artifact and evidence checkpoints."
        ),
        "source_manifests": (
            (
                f"source_{task_id}",
                digest(f"source_provenance|{task_id}|v1"),
            ),
        ),
        "administration_id": "administration_lab_closed_fixture_v1",
        "administration_manifest_digest": digest(
            "administration|lab_closed_fixture|v1"
        ),
        "stimulus_id": f"stimulus_{task_id}_v1",
        "stimulus_digest": digest(f"stimulus|{task_id}|v1"),
    }


def rubric(
    criterion_id: str,
    name: str,
    concept_id: str,
    dependence_group: str,
    *,
    evidence_cap: float,
    dependence_cap: float,
    misconception_id: str,
    assisted_factor: float = 0.20,
) -> RubricCriterion:
    return RubricCriterion(
        id=criterion_id,
        name=name,
        scale=CriterionScale.CONTINUOUS,
        concept_weights=((concept_id, 1.0),),
        dependence_group=dependence_group,
        misconception_ids=(misconception_id,),
        evidence_cap=evidence_cap,
        dependence_cap=dependence_cap,
        assisted_evidence_factor=assisted_factor,
    )


def build_tasks() -> dict[str, LearningTask]:
    implementation = LearningTask(
        id="task_cache_implementation",
        version=1,
        family_id="fam_cache_implementation",
        title="Implement a bounded cache from a pinned behavioral contract",
        modality=TaskModality.IMPLEMENTATION,
        **task_identity(
            "task_cache_implementation",
            "Implement a bounded cache from a pinned behavioral contract",
        ),
        scorer_contracts=(
            fixture_scorer_contract(
                "impl_contract", "impl_boundaries", "impl_structure"
            ),
        ),
        criteria=(
            rubric(
                "impl_contract",
                "Observable contract behavior",
                "c_implementation_contracts",
                "g_impl_behavior",
                evidence_cap=0.55,
                dependence_cap=0.70,
                misconception_id="m_happy_path_is_complete",
            ),
            rubric(
                "impl_boundaries",
                "Boundary and failure behavior",
                "c_boundary_reasoning",
                "g_impl_behavior",
                evidence_cap=0.55,
                dependence_cap=0.70,
                misconception_id="m_nominal_examples_prove_edges",
            ),
            rubric(
                "impl_structure",
                "Maintainable internal structure",
                "c_implementation_structure",
                "g_impl_structure",
                evidence_cap=0.30,
                dependence_cap=0.30,
                misconception_id="m_passing_output_implies_design_quality",
            ),
        ),
        allowed_tool_ids=("editor", "local_test_runner"),
        evidence_cap=0.85,
    )
    debugging = LearningTask(
        id="task_mask_debugging",
        version=1,
        family_id="fam_mask_debugging",
        title="Diagnose and repair a causal-mask regression",
        modality=TaskModality.DEBUGGING,
        **task_identity(
            "task_mask_debugging",
            "Diagnose and repair a causal-mask regression",
        ),
        scorer_contracts=(
            fixture_scorer_contract("debug_localize", "debug_repair"),
        ),
        criteria=(
            rubric(
                "debug_localize",
                "Localize the causal fault",
                "c_debug_fault_localization",
                "g_debug_diagnosis",
                evidence_cap=0.50,
                dependence_cap=0.50,
                misconception_id="m_symptom_location_is_cause",
                assisted_factor=0.25,
            ),
            rubric(
                "debug_repair",
                "Repair the invariant without regression",
                "c_debug_repair_validation",
                "g_debug_repair",
                evidence_cap=0.50,
                dependence_cap=0.50,
                misconception_id="m_one_passing_case_proves_repair",
                assisted_factor=0.25,
            ),
        ),
        allowed_tool_ids=("debugger", "editor", "local_test_runner"),
        evidence_cap=0.80,
    )
    explanation = LearningTask(
        id="task_attention_explanation",
        version=1,
        family_id="fam_attention_explanation",
        title="Explain why scaled dot-product attention uses its scale",
        modality=TaskModality.EXPLANATION,
        **task_identity(
            "task_attention_explanation",
            "Explain why scaled dot-product attention uses its scale",
        ),
        scorer_contracts=(
            fixture_scorer_contract("explain_mechanism", "explain_boundary"),
        ),
        criteria=(
            rubric(
                "explain_mechanism",
                "Connect variance growth to softmax behavior",
                "c_attention_scaling_mechanism",
                "g_explain_mechanism",
                evidence_cap=0.55,
                dependence_cap=0.55,
                misconception_id="m_scaling_is_only_convention",
            ),
            rubric(
                "explain_boundary",
                "State the limits of the argument",
                "c_explanation_scope",
                "g_explain_boundary",
                evidence_cap=0.35,
                dependence_cap=0.35,
                misconception_id="m_mechanism_claim_is_universal",
            ),
        ),
        allowed_tool_ids=("reference_viewer",),
        evidence_cap=0.75,
    )
    design = LearningTask(
        id="task_rag_design",
        version=1,
        family_id="fam_rag_design",
        title="Design an evidence-grounded retrieval pipeline",
        modality=TaskModality.DESIGN,
        **task_identity(
            "task_rag_design",
            "Design an evidence-grounded retrieval pipeline",
        ),
        scorer_contracts=(
            fixture_scorer_contract(
                "design_retrieval", "design_grounding", "design_evaluation"
            ),
        ),
        criteria=(
            rubric(
                "design_retrieval",
                "Retrieval and ranking choices",
                "c_rag_retrieval_design",
                "g_shared_design_artifact",
                evidence_cap=0.60,
                dependence_cap=0.70,
                misconception_id="m_more_chunks_always_improve_recall",
            ),
            rubric(
                "design_grounding",
                "Grounding and citation guarantees",
                "c_rag_grounding_design",
                "g_shared_design_artifact",
                evidence_cap=0.60,
                dependence_cap=0.70,
                misconception_id="m_retrieval_implies_grounding",
            ),
            rubric(
                "design_evaluation",
                "Failure-sensitive evaluation plan",
                "c_rag_evaluation_design",
                "g_shared_design_artifact",
                evidence_cap=0.60,
                dependence_cap=0.70,
                misconception_id="m_average_accuracy_covers_failures",
            ),
        ),
        allowed_tool_ids=("diagram_editor", "reference_viewer"),
        evidence_cap=0.90,
    )
    return {task.id: task for task in (implementation, debugging, explanation, design)}


@dataclass(frozen=True, slots=True)
class Scenario:
    id: str
    purpose: str
    task: LearningTask
    evaluation: TaskEvaluation
    actions: tuple[LearningAction, ...]
    bundle: EvidenceBundle
    observations: tuple[str, ...]
    comparisons: Mapping[str, Any]

    def terms(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "purpose": self.purpose,
            "task_id": self.task.id,
            "task_digest": self.task.digest,
            "evaluation": self.evaluation.terms(),
            "actions": [item.terms() for item in self.actions],
            "bundle": self.bundle.terms(),
            "bundle_digest": self.bundle.digest,
            "observations": list(self.observations),
            "comparisons": dict(self.comparisons),
        }


def scenario(
    scenario_id: str,
    purpose: str,
    task: LearningTask,
    evaluation: TaskEvaluation,
    actions: Sequence[LearningAction],
    observations: Sequence[str],
    comparisons: Mapping[str, Any] | None = None,
) -> Scenario:
    action_tuple = tuple(actions)
    return Scenario(
        id=scenario_id,
        purpose=purpose,
        task=task,
        evaluation=evaluation,
        actions=action_tuple,
        bundle=reduce_evidence(task, evaluation, action_tuple),
        observations=tuple(observations),
        comparisons={} if comparisons is None else comparisons,
    )


def build_clean_unassisted(task: LearningTask) -> Scenario:
    attempt = "att_clean_unassisted"
    actions = (
        action(attempt, 0, ActionKind.STARTED, {}, elapsed_ms=0),
        checkpoint(attempt, 1, "source_tree", elapsed_ms=38_000),
        check_run(attempt, 2, passed=12, failed=0, elapsed_ms=51_000),
        action(
            attempt,
            3,
            ActionKind.SUBMITTED,
            {"submission_digest": digest(f"{attempt}|submission")},
            elapsed_ms=62_000,
        ),
    )
    evaluations = tuple(
        criterion_evaluation(item.id, 0.95, (actions[1].id, actions[2].id))
        for item in task.criteria
    )
    return scenario(
        "clean_unassisted",
        "A reviewed implementation observation with no assistance.",
        task,
        task_evaluation(task, attempt, evaluations, actions),
        actions,
        (
            "Unassisted reviewed evidence remains certification-eligible.",
            "Dependence and task budgets still cap an otherwise clean result.",
        ),
    )


def build_hinted_assisted(task: LearningTask) -> Scenario:
    attempt = "att_hinted_assisted"
    actions = (
        action(attempt, 0, ActionKind.STARTED, {}, elapsed_ms=0),
        action(
            attempt,
            1,
            ActionKind.HINT_REQUESTED,
            {"hint_id": "hint_mask_axis", "level": 1},
            elapsed_ms=9_000,
        ),
        checkpoint(attempt, 2, "patch_digest", elapsed_ms=28_000),
        check_run(attempt, 3, passed=10, failed=0, elapsed_ms=37_000),
        action(
            attempt,
            4,
            ActionKind.SUBMITTED,
            {"submission_digest": digest(f"{attempt}|submission")},
            elapsed_ms=41_000,
        ),
    )
    evaluations = tuple(
        criterion_evaluation(item.id, 0.90, (actions[2].id, actions[3].id))
        for item in task.criteria
    )
    return scenario(
        "hinted_assisted",
        "A correct debugging repair follows a recorded hint.",
        task,
        task_evaluation(task, attempt, evaluations, actions),
        actions,
        (
            "The monotone phase reducer treats later work as assisted.",
            "Assisted success is visible but cannot certify unassisted competence.",
        ),
    )


def build_post_feedback(task: LearningTask) -> Scenario:
    attempt = "att_post_feedback"
    actions = (
        action(attempt, 0, ActionKind.STARTED, {}, elapsed_ms=0),
        action(
            attempt,
            1,
            ActionKind.SUBMITTED,
            {"submission_digest": digest(f"{attempt}|first_submission")},
            elapsed_ms=18_000,
        ),
        action(
            attempt,
            2,
            ActionKind.FEEDBACK_SHOWN,
            {"feedback_digest": digest(f"{attempt}|feedback")},
            phase=ActionPhase.POST_FEEDBACK,
            elapsed_ms=20_000,
        ),
        action(
            attempt,
            3,
            ActionKind.EXPLANATION_CHECKPOINT,
            {"explanation_digest": digest(f"{attempt}|explanation")},
            phase=ActionPhase.POST_FEEDBACK,
            elapsed_ms=31_000,
        ),
    )
    evaluations = tuple(
        criterion_evaluation(item.id, 1.0, (actions[3].id,))
        for item in task.criteria
    )
    return scenario(
        "post_feedback",
        "A polished explanation is recorded only after feedback was shown.",
        task,
        task_evaluation(task, attempt, evaluations, actions),
        actions,
        (
            "Declared phase cannot move backward after feedback.",
            "Post-feedback performance records learning activity, not prior mastery.",
        ),
    )


def build_missing_evaluation(task: LearningTask) -> Scenario:
    attempt = "att_missing_evaluation"
    actions = (
        action(attempt, 0, ActionKind.STARTED, {}, elapsed_ms=0),
        checkpoint(attempt, 1, "design_snapshot", elapsed_ms=46_000),
        action(
            attempt,
            2,
            ActionKind.SUBMITTED,
            {"submission_digest": digest(f"{attempt}|submission")},
            elapsed_ms=48_000,
        ),
    )
    return scenario(
        "missing_evaluation",
        "A design artifact exists but no rubric scorer result is available.",
        task,
        task_evaluation(task, attempt, (), actions),
        actions,
        (
            "Artifact presence alone is not competence evidence.",
            "Missing observations remain neutral rather than becoming failures.",
        ),
    )


def build_model_only(task: LearningTask) -> Scenario:
    attempt = "att_model_only"
    actions = (
        action(attempt, 0, ActionKind.STARTED, {}, elapsed_ms=0),
        action(
            attempt,
            1,
            ActionKind.EXPLANATION_CHECKPOINT,
            {"explanation_digest": digest(f"{attempt}|explanation")},
            elapsed_ms=24_000,
        ),
        action(
            attempt,
            2,
            ActionKind.SUBMITTED,
            {"submission_digest": digest(f"{attempt}|submission")},
            elapsed_ms=26_000,
        ),
    )
    evaluations = tuple(
        criterion_evaluation(
            item.id,
            0.92,
            (actions[1].id,),
            scorer_kind=ScorerKind.MODEL,
            scorer_id="shadow_model_fixture",
        )
        for item in task.criteria
    )
    return scenario(
        "model_only",
        "Only a quarantined model score exists for an explanation.",
        task,
        task_evaluation(task, attempt, evaluations, actions),
        actions,
        (
            "Model scores remain inspectable shadow data.",
            "A model cannot independently create mastery or certification evidence.",
        ),
    )


def build_restricted_tool(task: LearningTask) -> Scenario:
    attempt = "att_restricted_tool"
    actions = (
        action(attempt, 0, ActionKind.STARTED, {}, elapsed_ms=0),
        action(
            attempt,
            1,
            ActionKind.TOOL_USED,
            {"tool_id": "solution_retrieval", "purpose_code": "fetch_solution"},
            elapsed_ms=4_000,
        ),
        checkpoint(attempt, 2, "source_tree", elapsed_ms=21_000),
        check_run(attempt, 3, passed=12, failed=0, elapsed_ms=25_000),
        action(
            attempt,
            4,
            ActionKind.SUBMITTED,
            {"submission_digest": digest(f"{attempt}|submission")},
            elapsed_ms=27_000,
        ),
    )
    evaluations = tuple(
        criterion_evaluation(item.id, 0.95, (actions[2].id, actions[3].id))
        for item in task.criteria
    )
    return scenario(
        "restricted_tool_violation",
        "A high-scoring implementation violates a pinned administration condition.",
        task,
        task_evaluation(task, attempt, evaluations, actions),
        actions,
        (
            "The administration is invalidated without treating tool use as low skill.",
            "Scores remain visible, while evidence weight is zero and no penalty is emitted.",
        ),
    )


def build_local_dependence(task: LearningTask) -> Scenario:
    attempt = "att_local_dependence"
    actions = (
        action(attempt, 0, ActionKind.STARTED, {}, elapsed_ms=0),
        checkpoint(attempt, 1, "architecture_diagram", elapsed_ms=42_000),
        action(
            attempt,
            2,
            ActionKind.SUBMITTED,
            {"submission_digest": digest(f"{attempt}|submission")},
            elapsed_ms=45_000,
        ),
    )
    evaluations = tuple(
        criterion_evaluation(item.id, 0.90, (actions[1].id,))
        for item in task.criteria
    )
    return scenario(
        "locally_dependent_criteria",
        "Three rubric scores derive from the same design artifact.",
        task,
        task_evaluation(task, attempt, evaluations, actions),
        actions,
        (
            "Correlated rubric criteria share one finite information budget.",
            "Three labels on one artifact cannot masquerade as three independent trials.",
        ),
    )


def build_partial_checks(task: LearningTask) -> Scenario:
    attempt = "att_partial_checks"
    actions = (
        action(attempt, 0, ActionKind.STARTED, {}, elapsed_ms=0),
        checkpoint(attempt, 1, "source_tree", elapsed_ms=12_000),
        check_run(attempt, 2, passed=2, failed=6, elapsed_ms=18_000),
        action(
            attempt,
            3,
            ActionKind.ANSWER_REVISED,
            {"answer_digest": digest(f"{attempt}|revision|1")},
            elapsed_ms=25_000,
        ),
        check_run(attempt, 4, passed=6, failed=2, elapsed_ms=31_000),
        action(
            attempt,
            5,
            ActionKind.ANSWER_REVISED,
            {"answer_digest": digest(f"{attempt}|revision|2")},
            elapsed_ms=39_000,
        ),
        check_run(attempt, 6, passed=4, failed=4, elapsed_ms=44_000),
        checkpoint(attempt, 7, "source_tree", elapsed_ms=47_000),
        action(
            attempt,
            8,
            ActionKind.SUBMITTED,
            {"submission_digest": digest(f"{attempt}|submission")},
            elapsed_ms=49_000,
        ),
    )
    evaluations = tuple(
        criterion_evaluation(item.id, 0.55, (actions[7].id,), outcome_code="partial")
        for item in task.criteria
    )
    evaluation = task_evaluation(task, attempt, evaluations, actions)
    without_checks = resequence(
        item for item in actions if item.kind is not ActionKind.CHECK_RUN
    )
    comparison = reduce_evidence(
        task,
        task_evaluation(task, attempt, evaluations, without_checks),
        without_checks,
    )
    return scenario(
        "partial_checks_improve_then_regress",
        "Recorded test aggregates improve and then regress before submission.",
        task,
        evaluation,
        actions,
        (
            "The full check trajectory is retained rather than collapsed to the best run.",
            "Check telemetry does not manufacture rubric evidence without a scorer.",
        ),
        {
            "counterfactual_without_check_actions": {
                "bundle": comparison.terms(),
                "bundle_digest": comparison.digest,
            }
        },
    )


def build_allowed_tool_neutrality(task: LearningTask) -> Scenario:
    attempt = "att_allowed_tool"
    actions = (
        action(attempt, 0, ActionKind.STARTED, {}, elapsed_ms=0),
        action(
            attempt,
            1,
            ActionKind.TOOL_USED,
            {"tool_id": "debugger", "purpose_code": "inspect_state"},
            elapsed_ms=5_000,
        ),
        checkpoint(attempt, 2, "patch_digest", elapsed_ms=19_000),
        action(
            attempt,
            3,
            ActionKind.SUBMITTED,
            {"submission_digest": digest(f"{attempt}|submission")},
            elapsed_ms=23_000,
        ),
    )
    evaluations = tuple(
        criterion_evaluation(item.id, 0.80, (actions[2].id,))
        for item in task.criteria
    )
    evaluation = task_evaluation(task, attempt, evaluations, actions)
    without_tool = resequence(
        item for item in actions if item.kind is not ActionKind.TOOL_USED
    )
    comparison = reduce_evidence(
        task,
        task_evaluation(task, attempt, evaluations, without_tool),
        without_tool,
    )
    return scenario(
        "allowed_tool_neutrality",
        "A permitted debugger is observed during an otherwise identical attempt.",
        task,
        evaluation,
        actions,
        (
            "Allowed tool use is contextual telemetry, neither a bonus nor a penalty.",
            "The evidence provenance changes, but weights and scores do not.",
        ),
        {
            "counterfactual_without_tool_action": {
                "bundle": comparison.terms(),
                "bundle_digest": comparison.digest,
            }
        },
    )


def weight_signature(bundle: EvidenceBundle) -> tuple[Any, ...]:
    return (
        bundle.reported_task_score,
        bundle.evidence_score,
        bundle.total_evidence_weight,
        bundle.certification_evidence_weight,
        tuple(
            (
                record.criterion_id,
                record.score,
                record.effective_weight,
                record.certification_eligible,
            )
            for record in bundle.records
        ),
    )


def verify_scenarios(scenarios: Mapping[str, Scenario]) -> tuple[str, ...]:
    checks: list[str] = []

    clean_scenario = scenarios["clean_unassisted"]
    clean = clean_scenario.bundle
    require(clean.certification_evidence_weight > 0.0, "clean work did not certify")
    require(
        all(record.certification_eligible for record in clean.records),
        "clean unassisted records were not certification-eligible",
    )
    checks.append("clean_unassisted_certifies")

    committed = action_trace_digest(clean_scenario.actions)
    require(
        committed == clean_scenario.evaluation.action_trace_digest,
        "evaluation did not pin the full action trace",
    )
    require(
        committed == action_trace_digest(reversed(clean_scenario.actions)),
        "trace commitment depended on iterable order",
    )
    payload_mutation = tuple(
        replace(
            item,
            payload={
                **dict(item.payload),
                "artifact_digest": digest("adversarial_payload_mutation"),
            },
        )
        if item.kind is ActionKind.ARTIFACT_CHECKPOINT
        else item
        for item in clean_scenario.actions
    )
    timing_mutation = tuple(
        replace(item, elapsed_ms=(item.elapsed_ms or 0) + 1)
        if item.kind is ActionKind.ARTIFACT_CHECKPOINT
        else item
        for item in clean_scenario.actions
    )
    id_mutation = tuple(
        replace(item, id="act_clean_unassisted_mutated")
        if item.kind is ActionKind.ARTIFACT_CHECKPOINT
        else item
        for item in clean_scenario.actions
    )
    order_mutation = tuple(
        replace(item, sequence=2, elapsed_ms=51_000)
        if item.sequence == 1
        else replace(item, sequence=1, elapsed_ms=38_000)
        if item.sequence == 2
        else item
        for item in clean_scenario.actions
    )
    for label, mutation in (
        ("payload", payload_mutation),
        ("timing", timing_mutation),
        ("action_id", id_mutation),
        ("semantic_order", order_mutation),
    ):
        require(
            action_trace_digest(mutation) != committed,
            f"{label} mutation did not change full trace commitment",
        )
    require(
        summarize_actions(id_mutation).digest
        == summarize_actions(clean_scenario.actions).digest,
        "summary unexpectedly became a full action commitment",
    )
    checks.append("full_action_trace_commitment_is_exact")

    implementation_criterion = rubric(
        "mixed_implementation",
        "Executable implementation behavior",
        "c_mixed_implementation",
        "g_mixed_implementation",
        evidence_cap=0.50,
        dependence_cap=0.50,
        misconception_id="m_checks_prove_explanation",
    )
    explanation_criterion = rubric(
        "mixed_explanation",
        "Causal explanation quality",
        "c_mixed_explanation",
        "g_mixed_explanation",
        evidence_cap=0.50,
        dependence_cap=0.50,
        misconception_id="m_output_is_explanation",
    )
    scoped_task = LearningTask(
        id="task_mixed_scope_probe",
        version=1,
        family_id="fam_mixed_scope_probe",
        title="Implement and explain a reviewed behavior",
        modality=TaskModality.PROJECT,
        **task_identity(
            "task_mixed_scope_probe",
            "Implement and explain a reviewed behavior",
        ),
        criteria=(implementation_criterion, explanation_criterion),
        scorer_contracts=(
            ScorerContract(
                kind=ScorerKind.DETERMINISTIC,
                scorer_id="reviewed_fixture",
                scorer_version="v1",
                authority_id="authority_reviewed_fixture",
                authority_manifest_digest=digest(
                    "authority|reviewed_fixture|v1"
                ),
                criterion_ids=(implementation_criterion.id,),
                evidence_action_kinds=(ActionKind.CHECK_RUN,),
                check_set_manifests=(
                    (
                        "fixture_checks_v1",
                        digest("check_manifest|fixture_checks_v1"),
                    ),
                ),
            ),
        ),
    )
    scope_actions = (
        check_run("att_scope_probe", 0, passed=8, failed=0, elapsed_ms=5_000),
    )
    scope_bundle = reduce_evidence(
        scoped_task,
        task_evaluation(
            scoped_task,
            "att_scope_probe",
            (
                criterion_evaluation(
                    implementation_criterion.id,
                    1.0,
                    (scope_actions[0].id,),
                ),
                criterion_evaluation(
                    explanation_criterion.id,
                    1.0,
                    (scope_actions[0].id,),
                ),
            ),
            scope_actions,
        ),
        scope_actions,
    )
    scope_records = {record.criterion_id: record for record in scope_bundle.records}
    require(
        scope_records[implementation_criterion.id].effective_weight > 0.0,
        "authorized implementation check did not produce evidence",
    )
    explanation_record = scope_records[explanation_criterion.id]
    require(
        explanation_record.effective_weight == 0.0
        and "scorer_not_authorized_for_criterion"
        in explanation_record.reason_codes,
        "implementation checks cross-certified an explanation criterion",
    )
    wrong_scope_action = action(
        "att_wrong_check_set",
        0,
        ActionKind.CHECK_RUN,
        {
            "check_set_id": "unrelated_checks_v1",
            "passed": 8,
            "failed": 0,
            "errored": 0,
            "skipped": 0,
            "result_digest": digest("wrong_check_set"),
        },
        elapsed_ms=5_000,
    )
    wrong_scope_bundle = reduce_evidence(
        scoped_task,
        task_evaluation(
            scoped_task,
            "att_wrong_check_set",
            (
                criterion_evaluation(
                    implementation_criterion.id,
                    1.0,
                    (wrong_scope_action.id,),
                ),
            ),
            (wrong_scope_action,),
        ),
        (wrong_scope_action,),
    )
    require(
        wrong_scope_bundle.total_evidence_weight == 0.0
        and "unauthorized_check_set"
        in wrong_scope_bundle.records[0].reason_codes,
        "an unrelated check set crossed the scorer trust boundary",
    )
    checks.append("scorer_authority_is_criterion_and_source_scoped")

    assisted = scenarios["hinted_assisted"].bundle
    require(assisted.total_evidence_weight > 0.0, "assisted evidence disappeared")
    require(
        assisted.certification_evidence_weight == 0.0,
        "assisted work incorrectly certified unassisted competence",
    )
    require(
        all(record.phase is ActionPhase.ASSISTED for record in assisted.records),
        "post-hint phase did not remain assisted",
    )
    checks.append("assistance_is_visible_not_certifying")

    for scenario_id in ("post_feedback", "model_only", "missing_evaluation"):
        bundle = scenarios[scenario_id].bundle
        require(
            bundle.certification_evidence_weight == 0.0,
            f"{scenario_id} incorrectly produced certification evidence",
        )
        require(
            bundle.total_evidence_weight == 0.0,
            f"{scenario_id} incorrectly produced mastery evidence",
        )
    checks.append("post_feedback_model_and_missing_are_neutral")

    post_feedback = scenarios["post_feedback"].bundle
    require(
        not post_feedback.trace.phase_correction_action_ids,
        "valid post-feedback declarations were unexpectedly corrected",
    )
    require(
        all(record.phase is ActionPhase.POST_FEEDBACK for record in post_feedback.records),
        "feedback did not permanently advance the evidence phase",
    )
    checks.append("phase_is_monotone")

    missing = scenarios["missing_evaluation"].bundle
    require(
        missing.reported_task_score is None,
        "missing evaluation invented a reported task score",
    )
    require(missing.evidence_score is None, "missing evaluation invented evidence score")
    require(
        set(missing.missing_criterion_ids)
        == {record.criterion_id for record in missing.records},
        "missing criteria were not enumerated exactly",
    )
    checks.append("missing_is_not_failure")

    model = scenarios["model_only"].bundle
    require(
        model.reported_task_score is not None,
        "shadow score was not retained",
    )
    require(
        all("model_score_shadow_only" in record.reason_codes for record in model.records),
        "model-only reason was not preserved",
    )
    checks.append("model_score_is_shadow_only")

    restricted = scenarios["restricted_tool_violation"].bundle
    require(
        restricted.reported_task_score is not None,
        "restricted administration hid scores",
    )
    require(restricted.total_evidence_weight == 0.0, "restricted tool earned evidence")
    require(
        all(not record.misconception_ids for record in restricted.records),
        "tool policy violation was converted into a misconception",
    )
    require(
        all("task_condition_invalid" in record.reason_codes for record in restricted.records),
        "restricted condition was not explicitly invalidated",
    )
    checks.append("restricted_tool_invalidates_without_penalty")

    dependent = scenarios["locally_dependent_criteria"].bundle
    require(len(dependent.groups) == 1, "dependent criteria split across groups")
    group = dependent.groups[0]
    require(group.requested_weight > group.cap, "dependence cap was not challenged")
    require(group.group_capped_weight <= group.cap + 1e-12, "dependence cap failed")
    require(
        all("dependence_group_capped" in record.reason_codes for record in dependent.records),
        "record-level cap provenance is incomplete",
    )
    checks.append("local_dependence_has_one_budget")

    for current in scenarios.values():
        require(
            current.bundle.total_evidence_weight <= current.task.evidence_cap + 1e-12,
            f"{current.id} exceeded its task evidence cap",
        )
        for group_summary in current.bundle.groups:
            require(
                group_summary.effective_weight
                <= group_summary.group_capped_weight + 1e-12
                and group_summary.group_capped_weight <= group_summary.cap + 1e-12,
                f"{current.id} exceeded a dependence budget",
            )
        for record in current.bundle.records:
            require(
                0.0 <= record.effective_weight <= record.requested_weight + 1e-12,
                f"{current.id} has an invalid evidence weight",
            )
    checks.append("all_evidence_budgets_hold")

    partial = scenarios["partial_checks_improve_then_regress"]
    rates = tuple(item.pass_rate for item in partial.bundle.trace.test_progression)
    require(rates == (0.25, 0.75, 0.50), "check trajectory was changed or collapsed")
    require(rates[-1] < max(rates), "final regression was hidden")
    without_check_actions = resequence(
        item for item in partial.actions if item.kind is not ActionKind.CHECK_RUN
    )
    without_checks = reduce_evidence(
        partial.task,
        task_evaluation(
            partial.task,
            partial.evaluation.trace_id,
            partial.evaluation.criteria,
            without_check_actions,
        ),
        without_check_actions,
    )
    require(
        weight_signature(partial.bundle) == weight_signature(without_checks),
        "test telemetry directly changed rubric evidence",
    )
    checks.append("check_trajectory_is_observed_not_scored")

    allowed = scenarios["allowed_tool_neutrality"]
    without_tool_actions = resequence(
        item for item in allowed.actions if item.kind is not ActionKind.TOOL_USED
    )
    without_tool = reduce_evidence(
        allowed.task,
        task_evaluation(
            allowed.task,
            allowed.evaluation.trace_id,
            allowed.evaluation.criteria,
            without_tool_actions,
        ),
        without_tool_actions,
    )
    require(
        weight_signature(allowed.bundle) == weight_signature(without_tool),
        "allowed tool telemetry directly changed evidence",
    )
    checks.append("allowed_tool_telemetry_is_neutral")

    for current in scenarios.values():
        encoded = canonical_json(current.terms())
        decoded = json.loads(encoded, parse_constant=lambda value: require(False, value))
        require(isinstance(decoded, dict), f"{current.id} was not finite canonical JSON")
        require(
            current.bundle.digest == canonical_digest(current.bundle.terms()),
            f"{current.id} bundle digest is unstable",
        )
    checks.append("all_records_are_finite_and_hashable")

    for current in scenarios.values():
        decoded_task = LearningTask.from_terms(
            json.loads(canonical_json(current.task.terms()))
        )
        decoded_evaluation = TaskEvaluation.from_terms(
            json.loads(canonical_json(current.evaluation.terms()))
        )
        decoded_actions = tuple(
            LearningAction.from_terms(json.loads(canonical_json(item.terms())))
            for item in current.actions
        )
        require(
            decoded_task.digest == current.task.digest,
            f"{current.id} task terms did not round-trip",
        )
        require(
            decoded_evaluation.digest == current.evaluation.digest,
            f"{current.id} evaluation terms did not round-trip",
        )
        require(
            action_trace_digest(decoded_actions)
            == action_trace_digest(current.actions),
            f"{current.id} action terms did not round-trip",
        )
    checks.append("strict_persistence_terms_round_trip")
    return tuple(checks)


def build_payload() -> dict[str, Any]:
    tasks = build_tasks()
    scenarios = (
        build_clean_unassisted(tasks["task_cache_implementation"]),
        build_hinted_assisted(tasks["task_mask_debugging"]),
        build_post_feedback(tasks["task_attention_explanation"]),
        build_missing_evaluation(tasks["task_rag_design"]),
        build_model_only(tasks["task_attention_explanation"]),
        build_restricted_tool(tasks["task_cache_implementation"]),
        build_local_dependence(tasks["task_rag_design"]),
        build_partial_checks(tasks["task_cache_implementation"]),
        build_allowed_tool_neutrality(tasks["task_mask_debugging"]),
    )
    scenario_map = {item.id: item for item in scenarios}
    require(len(scenario_map) == len(scenarios), "scenario IDs are not unique")
    invariants = verify_scenarios(scenario_map)
    return {
        "schema": "tsq.multimodal_evidence_lab.v1",
        "lab_version": LAB_VERSION,
        "execution_boundary": {
            "artifact_execution": False,
            "learner_projection_updates": False,
            "model_calls": False,
            "raw_learner_content_retained": False,
            "scorer_outputs": "deterministic_recorded_fixtures_only",
            "telemetry_rule": "actions_are_context_not_competence_evidence",
        },
        "task_registry": {
            task_id: tasks[task_id].terms() for task_id in sorted(tasks)
        },
        "scenario_order": [item.id for item in scenarios],
        "scenarios": {item.id: item.terms() for item in scenarios},
        "verified_invariants": list(invariants),
    }


def run_lab() -> tuple[dict[str, Any], str]:
    first = build_payload()
    second = build_payload()
    first_json = canonical_json(first)
    require(first_json == canonical_json(second), "deterministic rerun diverged")
    payload = {
        **first,
        "deterministic_rerun": {
            "matched": True,
            "payload_digest": canonical_digest(first),
        },
    }
    encoded = canonical_json(payload)
    require(encoded == canonical_json(json.loads(encoded)), "canonical round-trip changed")
    return payload, encoded


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Ignored JSON artifact path (default: experiments/results/multimodal_evidence_lab.json).",
    )
    parser.add_argument(
        "--full-stdout",
        action="store_true",
        help="Print the full canonical artifact after the compact receipt.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload, encoded = run_lab()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded + "\n", encoding="utf-8")
    receipt = {
        "artifact_digest": canonical_digest(payload),
        "deterministic_rerun": True,
        "invariant_count": len(payload["verified_invariants"]),
        "output": str(output.relative_to(PROJECT_ROOT))
        if output.is_relative_to(PROJECT_ROOT)
        else str(output),
        "scenario_count": len(payload["scenarios"]),
        "status": "ok",
    }
    print(canonical_json(receipt))
    if args.full_stdout:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
