# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import math
import unittest

from tsq.evidence import (
    ActionKind,
    ActionPhase,
    CriterionEvaluation,
    CriterionScale,
    EvaluationStatus,
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


_D0 = "0" * 64
_D1 = "1" * 64
_D2 = "2" * 64
_D3 = "3" * 64


def action(
    sequence: int,
    kind: ActionKind,
    payload: dict[str, object],
    *,
    phase: ActionPhase = ActionPhase.UNASSISTED,
    elapsed_ms: int | None = None,
) -> LearningAction:
    return LearningAction(
        id=f"action_{sequence}",
        trace_id="attempt_1",
        sequence=sequence,
        kind=kind,
        phase=phase,
        payload=payload,
        elapsed_ms=elapsed_ms,
    )


def checkpoint(sequence: int = 0) -> LearningAction:
    return action(
        sequence,
        ActionKind.ARTIFACT_CHECKPOINT,
        {"artifact_digest": _D3, "artifact_kind": "reviewed_fixture"},
        elapsed_ms=sequence * 100,
    )


def criterion(
    criterion_id: str,
    *,
    group: str = "group_primary",
    group_cap: float = 1.0,
    evidence_cap: float = 1.0,
    assisted_factor: float = 0.0,
    concepts: tuple[tuple[str, float], ...] = (("concept_core", 1.0),),
    misconceptions: tuple[str, ...] = (),
) -> RubricCriterion:
    return RubricCriterion(
        id=criterion_id,
        name=criterion_id.replace("_", " ").title(),
        scale=CriterionScale.CONTINUOUS,
        concept_weights=concepts,
        dependence_group=group,
        misconception_ids=misconceptions,
        evidence_cap=evidence_cap,
        dependence_cap=group_cap,
        assisted_evidence_factor=assisted_factor,
    )


def task(
    *criteria: RubricCriterion,
    modality: TaskModality = TaskModality.IMPLEMENTATION,
    evidence_cap: float = 1.0,
    allowed_tools: tuple[str, ...] | None = None,
) -> LearningTask:
    return LearningTask(
        id="task_cache_debug",
        version=3,
        family_id="family_cache_debug",
        title="Diagnose and repair stale cache behavior",
        modality=modality,
        criteria=tuple(criteria),
        scorer_contracts=(
            ScorerContract(
                kind=ScorerKind.DETERMINISTIC,
                scorer_id="scorer_primary",
                scorer_version="v1",
                criterion_ids=tuple(item.id for item in criteria),
                evidence_action_kinds=(
                    ActionKind.ANSWER_REVISED,
                    ActionKind.ARTIFACT_CHECKPOINT,
                    ActionKind.EXPLANATION_CHECKPOINT,
                    ActionKind.CHECK_RUN,
                    ActionKind.SUBMITTED,
                ),
            ),
        ),
        allowed_tool_ids=allowed_tools,
        evidence_cap=evidence_cap,
    )


def evaluation(
    learning_task: LearningTask,
    *criteria: CriterionEvaluation,
    evaluation_id: str = "evaluation_1",
    actions: tuple[LearningAction, ...] = (),
) -> TaskEvaluation:
    return TaskEvaluation(
        id=evaluation_id,
        trace_id="attempt_1",
        task_id="task_cache_debug",
        task_version=3,
        task_digest=learning_task.digest,
        action_trace_digest=action_trace_digest(actions),
        criteria=tuple(criteria),
    )


def observed(
    criterion_id: str,
    score: float,
    *,
    phase: ActionPhase = ActionPhase.UNASSISTED,
    scorer: ScorerKind = ScorerKind.DETERMINISTIC,
    scorer_id: str = "scorer_primary",
    sources: tuple[str, ...] = (),
    misconceptions: tuple[str, ...] = (),
    reliability: float = 1.0,
) -> CriterionEvaluation:
    return CriterionEvaluation(
        criterion_id=criterion_id,
        status=EvaluationStatus.VALID,
        score=score,
        outcome_code="observed",
        phase=phase,
        scorer_kind=scorer,
        scorer_id=scorer_id,
        scorer_version="v1",
        source_action_ids=sources,
        misconception_ids=misconceptions,
        reliability=reliability,
    )


class LearningActionValidationTests(unittest.TestCase):
    def test_every_action_kind_has_an_exact_content_free_schema(self) -> None:
        payloads: dict[ActionKind, dict[str, object]] = {
            ActionKind.STARTED: {},
            ActionKind.HINT_REQUESTED: {"hint_id": "hint_cache_key", "level": 1},
            ActionKind.ANSWER_REVISED: {"answer_digest": _D0},
            ActionKind.ARTIFACT_CHECKPOINT: {
                "artifact_digest": _D1,
                "artifact_kind": "python_source",
            },
            ActionKind.EXPLANATION_CHECKPOINT: {"explanation_digest": _D2},
            ActionKind.CHECK_RUN: {
                "check_set_id": "cache_contract",
                "passed": 4,
                "failed": 1,
                "errored": 0,
                "skipped": 2,
                "result_digest": _D3,
            },
            ActionKind.TOOL_USED: {
                "tool_id": "python",
                "purpose_code": "run_checks",
            },
            ActionKind.SUBMITTED: {"submission_digest": _D0},
            ActionKind.FEEDBACK_SHOWN: {"feedback_digest": _D1},
            ActionKind.ABANDONED: {"reason_code": "time_expired"},
        }

        for sequence, (kind, payload) in enumerate(payloads.items()):
            with self.subTest(kind=kind):
                phase = (
                    ActionPhase.POST_FEEDBACK
                    if kind is ActionKind.FEEDBACK_SHOWN
                    else ActionPhase.UNASSISTED
                )
                accepted = action(sequence, kind, payload, phase=phase)
                self.assertEqual(dict(accepted.payload), payload)
                with self.assertRaisesRegex(ValueError, "unexpected raw_text"):
                    action(
                        sequence,
                        kind,
                        {**payload, "raw_text": "learner content"},
                        phase=phase,
                    )

    def test_digest_fields_raw_content_and_nonfinite_values_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            action(
                0,
                ActionKind.EXPLANATION_CHECKPOINT,
                {"explanation_digest": "my raw explanation"},
            )
        with self.assertRaisesRegex(ValueError, "unexpected explanation"):
            action(
                0,
                ActionKind.EXPLANATION_CHECKPOINT,
                {"explanation_digest": _D0, "explanation": "raw prose"},
            )
        with self.assertRaisesRegex(ValueError, "must be an integer between"):
            action(
                0,
                ActionKind.CHECK_RUN,
                {
                    "check_set_id": "checks",
                    "passed": math.nan,
                    "failed": 0,
                    "errored": 0,
                    "skipped": 0,
                    "result_digest": _D0,
                },
            )
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            canonical_json({"metric": math.inf})
        with self.assertRaisesRegex(ValueError, "between 0 and"):
            action(
                0,
                ActionKind.CHECK_RUN,
                {
                    "check_set_id": "checks",
                    "passed": 10**5000,
                    "failed": 0,
                    "errored": 0,
                    "skipped": 0,
                    "result_digest": _D0,
                },
            )

    def test_payload_is_copied_and_deeply_immutable(self) -> None:
        payload: dict[str, object] = {"answer_digest": _D0}
        recorded = action(0, ActionKind.ANSWER_REVISED, payload)
        payload["answer_digest"] = _D1
        self.assertEqual(recorded.payload["answer_digest"], _D0)
        with self.assertRaises(TypeError):
            recorded.payload["answer_digest"] = _D2  # type: ignore[index]


class ActionSummaryTests(unittest.TestCase):
    def test_full_trace_commitment_binds_every_action_term(self) -> None:
        base = (
            action(
                0,
                ActionKind.ARTIFACT_CHECKPOINT,
                {"artifact_digest": _D0, "artifact_kind": "python_source"},
            ),
            action(
                1,
                ActionKind.CHECK_RUN,
                {
                    "check_set_id": "implementation_checks",
                    "passed": 3,
                    "failed": 1,
                    "errored": 0,
                    "skipped": 0,
                    "result_digest": _D1,
                },
            ),
        )
        payload_mutation = (
            action(
                0,
                ActionKind.ARTIFACT_CHECKPOINT,
                {"artifact_digest": _D2, "artifact_kind": "python_source"},
            ),
            base[1],
        )
        timing_mutation = (
            action(
                0,
                ActionKind.ARTIFACT_CHECKPOINT,
                {"artifact_digest": _D0, "artifact_kind": "python_source"},
                elapsed_ms=1,
            ),
            base[1],
        )
        id_mutation = (
            LearningAction(
                id="renamed_action",
                trace_id=base[0].trace_id,
                sequence=base[0].sequence,
                kind=base[0].kind,
                phase=base[0].phase,
                payload=base[0].payload,
                elapsed_ms=base[0].elapsed_ms,
            ),
            base[1],
        )
        order_mutation = (
            LearningAction(
                id=base[0].id,
                trace_id=base[0].trace_id,
                sequence=1,
                kind=base[0].kind,
                phase=base[0].phase,
                payload=base[0].payload,
            ),
            LearningAction(
                id=base[1].id,
                trace_id=base[1].trace_id,
                sequence=0,
                kind=base[1].kind,
                phase=base[1].phase,
                payload=base[1].payload,
            ),
        )

        committed = action_trace_digest(base)
        self.assertEqual(committed, action_trace_digest(reversed(base)))
        for mutation in (
            payload_mutation,
            timing_mutation,
            id_mutation,
            order_mutation,
        ):
            with self.subTest(mutation=mutation):
                self.assertNotEqual(committed, action_trace_digest(mutation))

        # The summary is deliberately a lossy projection and cannot serve as
        # the evaluation's trace commitment.
        self.assertEqual(
            summarize_actions(base).digest,
            summarize_actions(id_mutation).digest,
        )

    def test_lifecycle_actions_fail_closed_when_duplicated_or_out_of_order(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be post_feedback"):
            action(
                0,
                ActionKind.FEEDBACK_SHOWN,
                {"feedback_digest": _D0},
            )
        with self.assertRaisesRegex(ValueError, "repeat singleton"):
            summarize_actions(
                (
                    action(0, ActionKind.SUBMITTED, {"submission_digest": _D0}),
                    action(1, ActionKind.SUBMITTED, {"submission_digest": _D1}),
                )
            )
        with self.assertRaisesRegex(ValueError, "first trace checkpoint"):
            summarize_actions(
                (
                    checkpoint(0),
                    action(1, ActionKind.STARTED, {}),
                )
            )
        with self.assertRaisesRegex(ValueError, "follow an abandoned"):
            summarize_actions(
                (
                    action(0, ActionKind.ABANDONED, {"reason_code": "left"}),
                    checkpoint(1),
                )
            )

    def test_trace_rejects_sequence_gaps_and_reversed_elapsed_time(self) -> None:
        with self.assertRaisesRegex(ValueError, "contiguous"):
            summarize_actions(
                (
                    action(0, ActionKind.STARTED, {}, elapsed_ms=0),
                    action(
                        2,
                        ActionKind.ANSWER_REVISED,
                        {"answer_digest": _D0},
                        elapsed_ms=20,
                    ),
                )
            )
        with self.assertRaisesRegex(ValueError, "elapsed times.*monotonic"):
            summarize_actions(
                (
                    action(0, ActionKind.STARTED, {}, elapsed_ms=20),
                    action(
                        1,
                        ActionKind.ANSWER_REVISED,
                        {"answer_digest": _D0},
                        elapsed_ms=10,
                    ),
                )
            )

    def test_trace_escalates_after_hint_and_feedback(self) -> None:
        actions = (
            action(0, ActionKind.STARTED, {}, elapsed_ms=0),
            action(
                1,
                ActionKind.HINT_REQUESTED,
                {"hint_id": "hint_cache_key", "level": 1},
                elapsed_ms=100,
            ),
            action(
                2,
                ActionKind.ANSWER_REVISED,
                {"answer_digest": _D0},
                elapsed_ms=300,
            ),
            action(
                3,
                ActionKind.FEEDBACK_SHOWN,
                {"feedback_digest": _D1},
                phase=ActionPhase.POST_FEEDBACK,
                elapsed_ms=400,
            ),
            action(
                4,
                ActionKind.EXPLANATION_CHECKPOINT,
                {"explanation_digest": _D2},
                elapsed_ms=900,
            ),
        )

        summary = summarize_actions(reversed(actions))

        self.assertEqual(summary.action_count, 5)
        self.assertEqual(summary.elapsed_ms, 900)
        self.assertEqual(summary.hint_count, 1)
        self.assertEqual(summary.answer_revision_count, 1)
        self.assertEqual(summary.unassisted_action_count, 2)
        self.assertEqual(summary.assisted_action_count, 1)
        self.assertEqual(summary.post_feedback_action_count, 2)
        self.assertEqual(summary.feedback_sequence, 3)
        self.assertEqual(summary.phase_correction_action_ids, ("action_2", "action_4"))
        self.assertEqual(summary.answer_digests, (_D0,))
        self.assertEqual(summary.explanation_digests, (_D2,))
        self.assertEqual(summary.digest, canonical_digest(summary.terms()))
        self.assertEqual(summary.digest, summarize_actions(actions).digest)

    def test_check_progression_is_ordered_and_digest_only(self) -> None:
        second = action(
            2,
            ActionKind.CHECK_RUN,
            {
                "check_set_id": "contract",
                "passed": 5,
                "failed": 0,
                "errored": 0,
                "skipped": 1,
                "result_digest": _D2,
            },
            elapsed_ms=200,
        )
        first = action(
            1,
            ActionKind.CHECK_RUN,
            {
                "check_set_id": "contract",
                "passed": 2,
                "failed": 2,
                "errored": 0,
                "skipped": 0,
                "result_digest": _D1,
            },
            elapsed_ms=100,
        )

        summary = summarize_actions((second, first))

        self.assertEqual(summary.check_run_count, 2)
        self.assertEqual(
            [progress.pass_rate for progress in summary.test_progression], [0.5, 1.0]
        )
        self.assertEqual(
            [progress.result_digest for progress in summary.test_progression],
            [_D1, _D2],
        )


class EvidenceReductionTests(unittest.TestCase):
    def test_valid_multiconcept_evidence_preserves_named_misconception(self) -> None:
        misconception_id = "mis_cache_key_equals_object_identity"
        rubric = criterion(
            "criterion_diagnosis",
            concepts=(("concept_cache_key", 0.7), ("concept_state", 0.3)),
            misconceptions=(misconception_id,),
        )
        learning_task = task(rubric, modality=TaskModality.DEBUGGING)
        actions = (checkpoint(),)
        scored = evaluation(
            learning_task,
            observed(
                rubric.id,
                0.4,
                sources=("action_0",),
                misconceptions=(misconception_id,),
                reliability=0.8,
            ),
            actions=actions,
        )

        bundle = reduce_evidence(learning_task, scored, actions)
        record = bundle.records[0]

        self.assertEqual(learning_task.modality, TaskModality.DEBUGGING)
        self.assertEqual(learning_task.concept_ids, ("concept_cache_key", "concept_state"))
        self.assertEqual(learning_task.misconception_ids, (misconception_id,))
        self.assertEqual(record.concept_weights, rubric.concept_weights)
        self.assertEqual(record.misconception_ids, (misconception_id,))
        self.assertAlmostEqual(record.potential_weight, 0.8)
        self.assertAlmostEqual(record.effective_weight, 0.8)
        self.assertTrue(record.certification_eligible)
        self.assertAlmostEqual(bundle.reported_task_score or -1.0, 0.4)
        self.assertAlmostEqual(bundle.evidence_score or -1.0, 0.4)

        with self.assertRaisesRegex(ValueError, "undeclared misconceptions"):
            reduce_evidence(
                learning_task,
                evaluation(
                    learning_task,
                    observed(
                        rubric.id,
                        0.4,
                        sources=("action_0",),
                        misconceptions=("mis_undeclared",),
                    ),
                    actions=actions,
                ),
                actions,
            )

    def test_missing_observations_are_neutral(self) -> None:
        first = criterion("criterion_artifact")
        second = criterion("criterion_explanation", group="group_explanation")
        learning_task = task(first, second)
        actions = (checkpoint(),)

        partial = reduce_evidence(
            learning_task,
            evaluation(
                learning_task,
                observed(first.id, 0.75, sources=("action_0",)),
                actions=actions,
            ),
            actions,
        )
        missing_record = next(
            record for record in partial.records if record.criterion_id == second.id
        )

        self.assertEqual(partial.missing_criterion_ids, (second.id,))
        self.assertIsNone(missing_record.score)
        self.assertEqual(missing_record.status, EvaluationStatus.MISSING)
        self.assertEqual(missing_record.effective_weight, 0.0)
        self.assertIn("missing_evaluation", missing_record.reason_codes)
        self.assertAlmostEqual(partial.reported_task_score or -1.0, 0.75)
        self.assertAlmostEqual(partial.evidence_score or -1.0, 0.75)

        entirely_missing = reduce_evidence(
            learning_task, evaluation(learning_task)
        )
        self.assertIsNone(entirely_missing.reported_task_score)
        self.assertIsNone(entirely_missing.evidence_score)
        self.assertEqual(entirely_missing.total_evidence_weight, 0.0)
        self.assertEqual(
            entirely_missing.missing_criterion_ids,
            (first.id, second.id),
        )

    def test_model_only_and_post_feedback_scores_cannot_update_competence(self) -> None:
        model_criterion = criterion("criterion_model")
        feedback_criterion = criterion(
            "criterion_feedback", group="group_feedback"
        )
        learning_task = task(model_criterion, feedback_criterion)
        bundle = reduce_evidence(
            learning_task,
            evaluation(
                learning_task,
                observed(model_criterion.id, 0.95, scorer=ScorerKind.MODEL),
                observed(
                    feedback_criterion.id,
                    1.0,
                    phase=ActionPhase.POST_FEEDBACK,
                ),
            ),
        )
        records = {record.criterion_id: record for record in bundle.records}

        self.assertEqual(bundle.total_evidence_weight, 0.0)
        self.assertEqual(bundle.certification_evidence_weight, 0.0)
        self.assertIsNone(bundle.evidence_score)
        self.assertAlmostEqual(bundle.reported_task_score or -1.0, 0.975)
        self.assertIn(
            "model_score_shadow_only", records[model_criterion.id].reason_codes
        )
        self.assertIn(
            "post_feedback_observation", records[feedback_criterion.id].reason_codes
        )
        self.assertFalse(records[model_criterion.id].certification_eligible)
        self.assertFalse(records[feedback_criterion.id].certification_eligible)

    def test_unadjudicated_imported_score_is_shadow_only(self) -> None:
        imported = criterion("criterion_imported")
        learning_task = task(imported)
        bundle = reduce_evidence(
            learning_task,
            evaluation(
                learning_task,
                observed(imported.id, 1.0, scorer=ScorerKind.IMPORTED)
            ),
        )

        self.assertEqual(bundle.total_evidence_weight, 0.0)
        self.assertEqual(bundle.certification_evidence_weight, 0.0)
        self.assertIsNone(bundle.evidence_score)
        self.assertEqual(bundle.reported_task_score, 1.0)
        self.assertIn(
            "imported_score_unadjudicated", bundle.records[0].reason_codes
        )
        self.assertFalse(bundle.records[0].certification_eligible)

    def test_source_phase_escalation_applies_assistance_factor_and_feedback_zero(self) -> None:
        assisted = criterion(
            "criterion_assisted", assisted_factor=0.25, evidence_cap=0.8
        )
        feedback = criterion(
            "criterion_after_feedback",
            group="group_feedback",
            assisted_factor=1.0,
        )
        actions = (
            action(
                0,
                ActionKind.HINT_REQUESTED,
                {"hint_id": "hint_one", "level": 1},
            ),
            action(1, ActionKind.ANSWER_REVISED, {"answer_digest": _D0}),
            action(
                2,
                ActionKind.SUBMITTED,
                {"submission_digest": _D2},
            ),
            action(
                3,
                ActionKind.FEEDBACK_SHOWN,
                {"feedback_digest": _D1},
                phase=ActionPhase.POST_FEEDBACK,
            ),
            action(
                4,
                ActionKind.EXPLANATION_CHECKPOINT,
                {"explanation_digest": _D3},
                phase=ActionPhase.POST_FEEDBACK,
            ),
        )
        learning_task = task(assisted, feedback)
        bundle = reduce_evidence(
            learning_task,
            evaluation(
                learning_task,
                observed(assisted.id, 0.8, sources=("action_1",)),
                observed(feedback.id, 1.0, sources=("action_4",)),
                actions=actions,
            ),
            actions,
        )
        records = {record.criterion_id: record for record in bundle.records}
        assisted_record = records[assisted.id]
        feedback_record = records[feedback.id]

        self.assertEqual(assisted_record.phase, ActionPhase.ASSISTED)
        self.assertAlmostEqual(assisted_record.potential_weight, 0.8)
        self.assertAlmostEqual(assisted_record.requested_weight, 0.2)
        self.assertAlmostEqual(assisted_record.effective_weight, 0.2)
        self.assertIn("source_phase_escalated", assisted_record.reason_codes)
        self.assertIn("assisted_observation", assisted_record.reason_codes)
        self.assertFalse(assisted_record.certification_eligible)

        self.assertEqual(feedback_record.phase, ActionPhase.POST_FEEDBACK)
        self.assertEqual(feedback_record.requested_weight, 0.0)
        self.assertIn("post_feedback_observation", feedback_record.reason_codes)

    def test_tool_use_is_neutral_unless_it_violates_a_restricted_condition(self) -> None:
        rubric = criterion("criterion_design")
        checkpoint_action = checkpoint()
        tool_action = action(
            1,
            ActionKind.TOOL_USED,
            {"tool_id": "python", "purpose_code": "inspect_state"},
        )
        base_task = task(rubric)
        base_actions = (checkpoint_action,)
        tool_actions = (checkpoint_action, tool_action)
        without_tool = reduce_evidence(
            base_task,
            evaluation(
                base_task,
                observed(rubric.id, 0.7, sources=("action_0",)),
                actions=base_actions,
            ),
            base_actions,
        )
        unrestricted = reduce_evidence(
            base_task,
            evaluation(
                base_task,
                observed(rubric.id, 0.7, sources=("action_0",)),
                actions=tool_actions,
            ),
            tool_actions,
        )
        allowed_task = task(rubric, allowed_tools=("python",))
        allowed = reduce_evidence(
            allowed_task,
            evaluation(
                allowed_task,
                observed(rubric.id, 0.7, sources=("action_0",)),
                actions=tool_actions,
            ),
            tool_actions,
        )

        self.assertEqual(unrestricted.trace.tool_use_count, 1)
        self.assertEqual(unrestricted.trace.tool_ids, ("python",))
        self.assertEqual(
            unrestricted.records[0].effective_weight,
            without_tool.records[0].effective_weight,
        )
        self.assertEqual(
            allowed.records[0].effective_weight,
            without_tool.records[0].effective_weight,
        )

        restricted_task = task(rubric, allowed_tools=("editor",))
        restricted = reduce_evidence(
            restricted_task,
            evaluation(
                restricted_task,
                observed(rubric.id, 0.7, sources=("action_0",)),
                actions=tool_actions,
            ),
            tool_actions,
        )
        self.assertEqual(restricted.records[0].effective_weight, 0.0)
        self.assertIn("task_condition_invalid", restricted.records[0].reason_codes)
        self.assertEqual(restricted.total_evidence_weight, 0.0)
        self.assertIsNone(restricted.evidence_score)

    def test_dependence_groups_and_task_share_finite_evidence_budgets(self) -> None:
        first = criterion("criterion_one", group="group_shared", group_cap=0.6)
        second = criterion("criterion_two", group="group_shared", group_cap=0.6)
        transfer = criterion(
            "criterion_transfer", group="group_transfer", group_cap=1.0
        )
        learning_task = task(first, second, transfer, evidence_cap=0.8)
        actions = (checkpoint(),)
        bundle = reduce_evidence(
            learning_task,
            evaluation(
                learning_task,
                observed(first.id, 0.2, sources=("action_0",)),
                observed(second.id, 0.8, sources=("action_0",)),
                observed(transfer.id, 1.0, sources=("action_0",)),
                actions=actions,
            ),
            actions,
        )
        records = {record.criterion_id: record for record in bundle.records}
        groups = {group.group_id: group for group in bundle.groups}

        self.assertAlmostEqual(records[first.id].effective_weight, 0.15)
        self.assertAlmostEqual(records[second.id].effective_weight, 0.15)
        self.assertAlmostEqual(records[transfer.id].effective_weight, 0.5)
        self.assertAlmostEqual(bundle.total_evidence_weight, 0.8)
        self.assertAlmostEqual(groups["group_shared"].requested_weight, 2.0)
        self.assertAlmostEqual(groups["group_shared"].group_capped_weight, 0.6)
        self.assertAlmostEqual(groups["group_shared"].effective_weight, 0.3)
        self.assertAlmostEqual(groups["group_transfer"].effective_weight, 0.5)
        self.assertIn("dependence_group_capped", records[first.id].reason_codes)
        self.assertIn("task_evidence_capped", records[first.id].reason_codes)
        self.assertIn("task_evidence_capped", records[transfer.id].reason_codes)

    def test_evaluation_pins_exact_task_and_action_trace_digests(self) -> None:
        rubric = criterion("criterion_pinned")
        learning_task = task(rubric)
        actions = (checkpoint(),)
        scored = evaluation(
            learning_task,
            observed(rubric.id, 0.8, sources=("action_0",)),
            actions=actions,
        )
        altered_task = task(
            criterion("criterion_pinned", evidence_cap=0.5)
        )
        with self.assertRaisesRegex(ValueError, "pinned.*LearningTask"):
            reduce_evidence(altered_task, scored, actions)

        altered_actions = (
            action(
                0,
                ActionKind.ARTIFACT_CHECKPOINT,
                {"artifact_digest": _D2, "artifact_kind": "reviewed_fixture"},
            ),
        )
        with self.assertRaisesRegex(ValueError, "semantic action trace"):
            reduce_evidence(learning_task, scored, altered_actions)

        self.assertEqual(scored.action_trace_digest, action_trace_digest(actions))
        self.assertEqual(
            reduce_evidence(learning_task, scored, actions).action_trace_digest,
            scored.action_trace_digest,
        )

    def test_scorer_scope_prevents_code_checks_certifying_explanation(self) -> None:
        implementation = criterion("criterion_implementation")
        explanation = criterion(
            "criterion_explanation", group="group_explanation"
        )
        learning_task = LearningTask(
            id="task_cache_debug",
            version=3,
            family_id="family_cache_debug",
            title="Implement and explain a cache contract",
            modality=TaskModality.IMPLEMENTATION,
            criteria=(implementation, explanation),
            scorer_contracts=(
                ScorerContract(
                    kind=ScorerKind.DETERMINISTIC,
                    scorer_id="scorer_primary",
                    scorer_version="v1",
                    criterion_ids=(implementation.id,),
                    evidence_action_kinds=(
                        ActionKind.ARTIFACT_CHECKPOINT,
                        ActionKind.CHECK_RUN,
                    ),
                    check_set_ids=("implementation_checks",),
                    artifact_kinds=("python_source",),
                ),
            ),
        )
        check_actions = (
            action(
                0,
                ActionKind.CHECK_RUN,
                {
                    "check_set_id": "implementation_checks",
                    "passed": 4,
                    "failed": 0,
                    "errored": 0,
                    "skipped": 0,
                    "result_digest": _D0,
                },
            ),
        )
        bundle = reduce_evidence(
            learning_task,
            evaluation(
                learning_task,
                observed(implementation.id, 0.9, sources=("action_0",)),
                observed(explanation.id, 0.9, sources=("action_0",)),
                actions=check_actions,
            ),
            check_actions,
        )
        records = {record.criterion_id: record for record in bundle.records}
        self.assertGreater(records[implementation.id].effective_weight, 0.0)
        self.assertEqual(records[explanation.id].effective_weight, 0.0)
        self.assertIn(
            "scorer_not_authorized_for_criterion",
            records[explanation.id].reason_codes,
        )

        wrong_check_actions = (
            action(
                0,
                ActionKind.CHECK_RUN,
                {
                    "check_set_id": "explanation_style_checks",
                    "passed": 4,
                    "failed": 0,
                    "errored": 0,
                    "skipped": 0,
                    "result_digest": _D1,
                },
            ),
        )
        wrong_check = reduce_evidence(
            learning_task,
            evaluation(
                learning_task,
                observed(implementation.id, 0.9, sources=("action_0",)),
                actions=wrong_check_actions,
            ),
            wrong_check_actions,
        )
        self.assertEqual(wrong_check.total_evidence_weight, 0.0)
        self.assertIn("unauthorized_check_set", wrong_check.records[0].reason_codes)

        wrong_artifact_actions = (
            action(
                0,
                ActionKind.ARTIFACT_CHECKPOINT,
                {"artifact_digest": _D2, "artifact_kind": "design_diagram"},
            ),
        )
        wrong_artifact = reduce_evidence(
            learning_task,
            evaluation(
                learning_task,
                observed(implementation.id, 0.9, sources=("action_0",)),
                actions=wrong_artifact_actions,
            ),
            wrong_artifact_actions,
        )
        self.assertEqual(wrong_artifact.total_evidence_weight, 0.0)
        self.assertIn(
            "unauthorized_artifact_kind",
            wrong_artifact.records[0].reason_codes,
        )

    def test_unregistered_scorer_and_administrative_source_are_shadow_only(self) -> None:
        rubric = criterion("criterion_trust")
        learning_task = task(rubric)
        actions = (checkpoint(),)
        unknown = reduce_evidence(
            learning_task,
            evaluation(
                learning_task,
                observed(
                    rubric.id,
                    0.9,
                    scorer_id="unregistered_scorer",
                    sources=("action_0",),
                ),
                actions=actions,
            ),
            actions,
        )
        self.assertEqual(unknown.total_evidence_weight, 0.0)
        self.assertIn("untrusted_scorer", unknown.records[0].reason_codes)

        administrative = (
            action(0, ActionKind.STARTED, {}),
        )
        admin_source = reduce_evidence(
            learning_task,
            evaluation(
                learning_task,
                observed(rubric.id, 0.9, sources=("action_0",)),
                actions=administrative,
            ),
            administrative,
        )
        self.assertEqual(admin_source.total_evidence_weight, 0.0)
        self.assertIn(
            "non_evidence_source_action", admin_source.records[0].reason_codes
        )

    def test_human_scorer_requires_external_attestation(self) -> None:
        rubric = criterion("criterion_human")
        learning_task = LearningTask(
            id="task_cache_debug",
            version=3,
            family_id="family_cache_debug",
            title="Diagnose and repair stale cache behavior",
            modality=TaskModality.DEBUGGING,
            criteria=(rubric,),
            scorer_contracts=(
                ScorerContract(
                    kind=ScorerKind.HUMAN,
                    scorer_id="reviewer_one",
                    scorer_version="v1",
                    criterion_ids=(rubric.id,),
                    evidence_action_kinds=(ActionKind.ARTIFACT_CHECKPOINT,),
                    requires_attestation=True,
                ),
            ),
        )
        actions = (checkpoint(),)
        bundle = reduce_evidence(
            learning_task,
            evaluation(
                learning_task,
                observed(
                    rubric.id,
                    0.8,
                    scorer=ScorerKind.HUMAN,
                    scorer_id="reviewer_one",
                    sources=("action_0",),
                ),
                actions=actions,
            ),
            actions,
        )
        self.assertEqual(bundle.total_evidence_weight, 0.0)
        self.assertIn(
            "missing_scorer_attestation", bundle.records[0].reason_codes
        )

    def test_discrete_scale_rejects_undeclared_fractional_score(self) -> None:
        rubric = RubricCriterion(
            id="criterion_binary",
            name="Binary contract",
            scale=CriterionScale.BINARY,
            concept_weights=(("concept_core", 1.0),),
            dependence_group="group_binary",
        )
        learning_task = task(rubric)
        actions = (checkpoint(),)
        scored = evaluation(
            learning_task,
            observed(rubric.id, 0.5, sources=("action_0",)),
            actions=actions,
        )
        with self.assertRaisesRegex(ValueError, "binary scale"):
            reduce_evidence(learning_task, scored, actions)

    def test_restricted_tool_after_submission_does_not_invalidate_work(self) -> None:
        rubric = criterion("criterion_closed_window")
        learning_task = task(rubric, allowed_tools=("editor",))
        actions = (
            checkpoint(),
            action(1, ActionKind.SUBMITTED, {"submission_digest": _D1}),
            action(
                2,
                ActionKind.TOOL_USED,
                {"tool_id": "python", "purpose_code": "postmortem"},
                phase=ActionPhase.POST_FEEDBACK,
            ),
        )
        bundle = reduce_evidence(
            learning_task,
            evaluation(
                learning_task,
                observed(rubric.id, 0.8, sources=("action_0",)),
                actions=actions,
            ),
            actions,
        )
        self.assertGreater(bundle.total_evidence_weight, 0.0)
        self.assertNotIn("task_condition_invalid", bundle.records[0].reason_codes)

    def test_reduction_is_deterministic_for_replayed_action_order(self) -> None:
        rubric = criterion("criterion_replay", assisted_factor=0.5)
        learning_task = task(rubric, modality=TaskModality.TRANSFER)
        actions = (
            action(0, ActionKind.STARTED, {}),
            action(
                1,
                ActionKind.HINT_REQUESTED,
                {"hint_id": "hint_transfer", "level": 1},
            ),
            action(2, ActionKind.ARTIFACT_CHECKPOINT, {
                "artifact_digest": _D3,
                "artifact_kind": "design_diagram",
            }),
        )
        scored = evaluation(
            learning_task,
            observed(rubric.id, 0.6, sources=("action_2",)),
            actions=actions,
        )

        forward = reduce_evidence(learning_task, scored, actions)
        replayed = reduce_evidence(learning_task, scored, reversed(actions))

        self.assertEqual(forward.terms(), replayed.terms())
        self.assertEqual(forward.digest, replayed.digest)
        self.assertEqual(forward.task_digest, learning_task.digest)
        self.assertEqual(forward.evaluation_digest, scored.digest)
        self.assertEqual(forward.records[0].id, replayed.records[0].id)
        self.assertEqual(
            forward.records[0].provenance_digest,
            replayed.records[0].provenance_digest,
        )


if __name__ == "__main__":
    unittest.main()
