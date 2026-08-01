# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import copy
import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from tsq.capacity import VERIFICATION_KINDS
from tsq.corpus import load_bundle, parse_bundle, parse_catalog, read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.errors import ConflictError, ExhaustedError, ValidationError
from tsq.learner import (
    CONCEPT_MODEL_VERSION,
    MODEL_VERSION,
    OBJECTIVE_GAUSSIAN_MODEL_VERSION,
    SESSION_LAPSE_RATE,
    LearnerModel,
)
from tsq.models import (
    ConceptRole,
    ConceptWeight,
    LearningObjective,
    ObjectiveOperation,
    ObjectiveState,
    Option,
    Question,
    QuestionKind,
    QuestionStatus,
    SessionPhase,
    SkillState,
    sigmoid,
)
from tsq.replay import ProjectionReplay
from tsq.store import Database
from tsq.versions import BOUND_QUESTION_SELECTED_EVENT_SCHEMA_VERSION


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
START = datetime(2100, 7, 8, 9, 0, tzinfo=timezone.utc)


def declared_fixture_bundle(bundle: dict) -> dict:
    """Give intentionally edited fixture questions explicit provenance."""

    declared = copy.deepcopy(bundle)
    for question in declared["questions"]:
        question.setdefault("provenance", {}).setdefault("generated", False)
    return declared


def legacy_bundle(bundle: dict) -> dict:
    """Remove the optional v2 extension from an explicitly declared fixture."""

    legacy = declared_fixture_bundle(bundle)
    legacy["schema_version"] = 1
    legacy.pop("learning_objectives", None)
    legacy.pop("objective_edges", None)
    for question in legacy["questions"]:
        question.pop("learning_objective_id", None)
        for option in question["options"]:
            option.pop("diagnostic_objective_id", None)
    return legacy


class ObjectiveCorpusSchemaTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_bundle(CORPUS)

    def test_v2_hydrates_direct_and_diagnostic_objective_mappings(self) -> None:
        questions = parse_bundle(copy.deepcopy(self.bundle))[4]
        by_id = {question.id: question for question in questions}
        raw = next(
            question
            for question in self.bundle["questions"]
            if question.get("learning_objective_id")
        )
        question = by_id[raw["id"]]

        self.assertIsNotNone(question.objective)
        self.assertEqual(question.objective_id, raw["learning_objective_id"])
        self.assertIn(question.primary_concept_id, question.objective.concept_ids)
        for raw_option, option in zip(raw["options"], question.options, strict=True):
            expected = raw_option.get("diagnostic_objective_id")
            if expected is None and not raw_option["correct"]:
                expected = raw["learning_objective_id"]
            self.assertEqual(option.diagnostic_objective_id, expected)

    def test_v3_hydrates_explicit_directed_objective_prerequisites(self) -> None:
        questions = parse_bundle(copy.deepcopy(self.bundle))[4]
        objectives = {
            question.objective.id: question.objective
            for question in questions
            if question.objective is not None
        }

        self.assertEqual(
            {objective.objective_graph_version for objective in objectives.values()},
            {1},
        )
        dependent = objectives["lo_incremental_kv_cache"]
        self.assertEqual(
            {edge.source_id for edge in dependent.prerequisites},
            {"lo_causal_visibility", "lo_transformer_information_paths"},
        )
        self.assertTrue(
            all(edge.target_id == dependent.id for edge in dependent.prerequisites)
        )
        self.assertTrue(all(edge.rationale.strip() for edge in dependent.prerequisites))

    def test_v3_rejects_objective_prerequisite_cycles(self) -> None:
        cyclic = copy.deepcopy(self.bundle)
        cyclic["objective_edges"].append(
            {
                "id": "oe_order_cycle_value",
                "source": "lo_attention_permutation_order",
                "target": "lo_attention_value_routing",
                "relation": "prerequisite",
                "weight": 0.8,
                "rationale": "Adversarial reverse edge used only to test cycle rejection.",
            }
        )

        with self.assertRaises(ValidationError) as raised:
            parse_bundle(cyclic)

        self.assertIn(
            "objective_prerequisite_cycle",
            {issue.code for issue in raised.exception.issues},
        )

    def test_v2_rejects_unknown_and_semantically_incompatible_mappings(self) -> None:
        unknown = declared_fixture_bundle(self.bundle)
        mapped = next(
            question
            for question in unknown["questions"]
            if question.get("learning_objective_id")
        )
        mapped["learning_objective_id"] = "lo_missing_test_objective"
        with self.assertRaises(ValidationError) as raised:
            parse_bundle(unknown)
        self.assertIn(
            "unknown_learning_objective",
            {issue.code for issue in raised.exception.issues},
        )

        incompatible = declared_fixture_bundle(self.bundle)
        objective_concepts = {
            objective["id"]: {
                objective["primary_concept_id"],
                *objective.get("supporting_concept_ids", []),
            }
            for objective in incompatible["learning_objectives"]
        }
        misconception_owners = {
            misconception["id"]: misconception["concept_id"]
            for misconception in incompatible["misconceptions"]
        }
        incompatible_option = None
        incompatible_objective_id = None
        for question in incompatible["questions"]:
            for option in question["options"]:
                misconception_id = option.get("misconception_id")
                if option["correct"] or misconception_id is None:
                    continue
                owner = misconception_owners[misconception_id]
                incompatible_objective_id = next(
                    (
                        objective_id
                        for objective_id, concept_ids in objective_concepts.items()
                        if owner not in concept_ids
                    ),
                    None,
                )
                if incompatible_objective_id is not None:
                    incompatible_option = option
                    break
            if incompatible_option is not None:
                break
        self.assertIsNotNone(incompatible_option)
        self.assertIsNotNone(incompatible_objective_id)
        incompatible_option["diagnostic_objective_id"] = incompatible_objective_id

        with self.assertRaises(ValidationError) as raised:
            parse_bundle(incompatible)
        self.assertIn(
            "diagnostic_objective_owner_mismatch",
            {issue.code for issue in raised.exception.issues},
        )

    def test_schema_v1_remains_objective_free_and_compatible(self) -> None:
        legacy = legacy_bundle(self.bundle)
        questions = parse_bundle(legacy)[4]

        self.assertEqual(len(questions), len(self.bundle["questions"]))
        self.assertTrue(all(question.objective is None for question in questions))
        self.assertTrue(
            all(
                option.diagnostic_objective_id is None
                for question in questions
                for option in question.options
            )
        )


class ObjectiveLearnerMathTestCase(unittest.TestCase):
    def test_information_gain_uses_objective_probability_in_denominator(self) -> None:
        objective = LearningObjective(
            id="lo_information_math",
            name="Predict the modeled outcome",
            description="Predict the outcome represented by the objective latent state.",
            primary_concept_id="c_information_math",
            supporting_concept_ids=(),
            operation=ObjectiveOperation.PREDICT,
        )
        question = Question(
            id="q_information_math",
            version=1,
            family_id="f_information_math",
            status=QuestionStatus.CALIBRATED,
            stem="Which outcome follows from the modeled objective state?",
            kind=QuestionKind.CONCEPTUAL,
            difficulty=0.4,
            discrimination=1.7,
            guess_rate=0.25,
            slip_rate=0.05,
            concepts=(
                ConceptWeight(
                    "c_information_math", 1.0, ConceptRole.PRIMARY
                ),
            ),
            options=(
                Option("a", "Outcome A", True, "This is the modeled outcome."),
                Option("b", "Outcome B", False, "This reverses the outcome."),
                Option("c", "Outcome C", False, "This ignores the condition."),
                Option("d", "Outcome D", False, "This adds an unsupported premise."),
            ),
            source_ids=("src_information_math",),
            objective=objective,
        )
        concept_state = SkillState(
            learner_id="learner",
            concept_id="c_information_math",
            mean=-3.0,
            variance=0.6,
            stability_hours=48.0,
        )
        objective_state = ObjectiveState(
            learner_id="learner",
            objective_id=objective.id,
            mean=2.0,
            variance=1.4,
            stability_hours=48.0,
        )
        states = {concept_state.concept_id: concept_state}
        # This is a byte-compatibility regression for the retired v5 Fisher
        # approximation; v6 uses full-posterior expected information instead.
        model = LearnerModel(OBJECTIVE_GAUSSIAN_MODEL_VERSION)

        objective_probability = model.predict_correct(
            question, states, objective_state=objective_state
        )
        logistic = sigmoid(
            question.discrimination
            * (objective_state.mean - question.difficulty)
        )
        derivative = (
            (1.0 - SESSION_LAPSE_RATE)
            * (1.0 - question.guess_rate - question.slip_rate)
            * question.discrimination
            * logistic
            * (1.0 - logistic)
        )
        objective_fisher = derivative**2 / (
            objective_probability * (1.0 - objective_probability)
        )
        expected = objective_state.variance - 1.0 / (
            1.0 / objective_state.variance + objective_fisher
        )
        stale_concept_logistic = sigmoid(
            question.discrimination
            * (concept_state.mean - question.difficulty)
        )
        stale_concept_probability = (
            SESSION_LAPSE_RATE / len(question.options)
            + (1.0 - SESSION_LAPSE_RATE)
            * (
                question.guess_rate
                + (1.0 - question.guess_rate - question.slip_rate)
                * stale_concept_logistic
            )
        )
        stale_concept_denominator = objective_state.variance - 1.0 / (
            1.0 / objective_state.variance
            + derivative**2
            / (
                stale_concept_probability
                * (1.0 - stale_concept_probability)
            )
        )

        with patch.object(
            model, "predict_correct", wraps=model.predict_correct
        ) as predict_correct:
            actual = model.expected_information_gain(
                question, states, objective_state=objective_state
            )

        predict_correct.assert_called_once_with(
            question, states, objective_state=objective_state
        )
        self.assertAlmostEqual(actual, expected, places=12)
        self.assertNotAlmostEqual(
            actual, stale_concept_denominator, places=8
        )

    def test_objective_prediction_is_isolated_from_all_concept_latents(self) -> None:
        objective = LearningObjective(
            id="lo_direct_only_math",
            name="Apply one direct objective",
            description="Apply the directly assessed objective without proxy evidence.",
            primary_concept_id="c_broad_primary",
            supporting_concept_ids=("c_secondary_context",),
            operation=ObjectiveOperation.APPLY,
        )
        question = Question(
            id="q_direct_only_math",
            version=1,
            family_id="f_direct_only_math",
            status=QuestionStatus.CALIBRATED,
            stem="Which outcome follows from the directly assessed objective?",
            kind=QuestionKind.APPLICATION,
            difficulty=0.2,
            discrimination=1.4,
            guess_rate=0.25,
            slip_rate=0.05,
            concepts=(
                ConceptWeight("c_broad_primary", 0.75, ConceptRole.PRIMARY),
                ConceptWeight("c_secondary_context", 0.25, ConceptRole.SECONDARY),
            ),
            options=(
                Option("a", "Outcome A", True, "This follows from the objective."),
                Option("b", "Outcome B", False, "This reverses the objective."),
                Option("c", "Outcome C", False, "This ignores the condition."),
                Option("d", "Outcome D", False, "This adds a premise."),
            ),
            source_ids=("src_direct_only_math",),
            objective=objective,
        )
        objective_state = ObjectiveState(
            "learner",
            objective.id,
            mean=0.6,
            variance=1.1,
            stability_hours=48.0,
        )

        def concept_states(mean: float) -> dict[str, SkillState]:
            return {
                concept_id: SkillState(
                    "learner",
                    concept_id,
                    mean=mean,
                    variance=0.7,
                    stability_hours=48.0,
                )
                for concept_id in ("c_broad_primary", "c_secondary_context")
            }

        model = LearnerModel()
        low_states = concept_states(-5.0)
        high_states = concept_states(5.0)
        direct_only = replace(
            question,
            concepts=(
                ConceptWeight("c_broad_primary", 1.0, ConceptRole.PRIMARY),
            ),
        )

        low_prediction = model.predict_correct(
            question, low_states, objective_state=objective_state
        )
        high_prediction = model.predict_correct(
            question, high_states, objective_state=objective_state
        )
        direct_prediction = model.predict_correct(
            direct_only, low_states, objective_state=objective_state
        )
        low_information = model.expected_information_gain(
            question, low_states, objective_state=objective_state
        )
        high_information = model.expected_information_gain(
            question, high_states, objective_state=objective_state
        )
        direct_information = model.expected_information_gain(
            direct_only, low_states, objective_state=objective_state
        )

        self.assertAlmostEqual(low_prediction, high_prediction, places=12)
        self.assertAlmostEqual(low_prediction, direct_prediction, places=12)
        self.assertAlmostEqual(low_information, high_information, places=12)
        self.assertAlmostEqual(low_information, direct_information, places=12)


class ObjectiveRuntimeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.tempdir.name) / "objective.db")
        self.database.initialize()
        parsed = read_and_parse(CORPUS, include_catalog=True)
        self.questions = parsed[4]
        self.release_id = self.database.import_corpus(*parsed)["release_id"]
        self.engine = AdaptiveEngine(self.database)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _serviceable_objective_trigger(self):
        families_by_diagnosis: dict[tuple[str, str], set[str]] = {}
        for question in self.questions:
            if not question.status.eligible_for_adaptation or not question.objective_id:
                continue
            for misconception_id in question.misconception_ids:
                families_by_diagnosis.setdefault(
                    (question.objective_id, misconception_id), set()
                ).add(question.family_id)

        for seed in range(64):
            learner_id = f"objective-routing-{seed}"
            self.engine.create_learner(learner_id)
            session = self.engine.start_session(
                learner_id, "t_transformers", mode="learn", seed=seed
            )
            presentation = self.engine.next_question(
                session["id"], now=START + timedelta(seconds=seed)
            )
            question = presentation.question
            for option in question.options:
                if option.correct or option.misconception_id is None:
                    continue
                target = option.diagnostic_objective_id or question.objective_id
                if target is not None and len(
                    families_by_diagnosis.get(
                        (target, option.misconception_id), set()
                    )
                ) >= 3:
                    return learner_id, session, presentation, option, target
        self.fail("No objective diagnosis had three independent live families.")

    def test_runtime_rejects_a_coherently_tampered_focus_tuple(self) -> None:
        learner_id = "tampered-objective-focus"
        self.engine.create_learner(learner_id)
        session = self.engine.start_session(
            learner_id, "t_transformers", mode="learn", seed=0
        )
        presentation = self.engine.next_question(session["id"], now=START)
        objective = next(
            objective
            for objective in self.database.get_learning_objectives(
                self.release_id
            )
            if objective.primary_concept_id
            != presentation.question.primary_concept_id
        )
        wrong_owner = next(
            concept_id
            for concept_id in self.database.get_graph(self.release_id).concepts
            if concept_id != objective.primary_concept_id
        )
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE sessions
                   SET phase = 'remediate', focus_concept_id = ?,
                       focus_misconception_id = NULL,
                       focus_objective_id = ?
                   WHERE id = ?""",
                (wrong_owner, objective.id, session["id"]),
            )
            connection.execute(
                """UPDATE decisions
                   SET phase = 'remediate', focus_concept_id = ?,
                       focus_misconception_id = NULL,
                       focus_objective_id = ?
                   WHERE id = ?""",
                (wrong_owner, objective.id, presentation.decision_id),
            )

        with self.assertRaisesRegex(ValidationError, "canonical owner"):
            self.engine.next_question(
                session["id"], now=START + timedelta(minutes=1)
            )
        with self.assertRaisesRegex(ValidationError, "canonical owner"):
            self.engine.record_action(
                presentation.decision_id,
                "answer_changed",
                {"change_count": 1},
                now=START + timedelta(minutes=1),
            )
        with self.assertRaisesRegex(ValidationError, "canonical owner"):
            self.engine.submit_answer(
                presentation.decision_id,
                presentation.question.correct_option.id,
                now=START + timedelta(minutes=1),
            )

        integrity = self.database.verify_integrity()
        self.assertFalse(integrity["ok"])
        self.assertTrue(
            any("canonical owner" in error for error in integrity["errors"]),
            integrity["errors"],
        )

    def test_objective_evidence_replaces_broad_primary_and_routes_exactly(self) -> None:
        learner_id, session, trigger, wrong, target_objective = (
            self._serviceable_objective_trigger()
        )
        trigger_question = trigger.question
        answered_at = START + timedelta(minutes=2)

        diagnosed = self.engine.submit_answer(
            trigger.decision_id,
            wrong.id,
            confidence=0.9,
            response_ms=1200,
            idempotency_key="objective-diagnosis",
            now=answered_at,
        )

        self.assertEqual(diagnosed.next_phase, SessionPhase.REMEDIATE)
        self.assertEqual(diagnosed.focus_objective_id, target_objective)
        self.assertEqual(diagnosed.focus_misconception_id, wrong.misconception_id)
        # The response is evidence about the item's direct objective.  The
        # selected misconception may route repair to a different, diagnostic
        # objective, but must not rewrite what the item itself measured.
        self.assertIn(
            trigger_question.objective_id,
            self.database.get_objective_states(learner_id),
        )
        self.assertNotIn(
            trigger_question.primary_concept_id,
            self.database.get_skill_states(learner_id),
        )
        self.assertTrue(
            any(
                change.get("objective_id") == trigger_question.objective_id
                for change in diagnosed.state_changes
            )
        )
        self.assertFalse(
            any(
                change.get("concept_id") == trigger_question.primary_concept_id
                for change in diagnosed.state_changes
            )
        )

        repair = self.engine.next_question(
            session["id"], now=answered_at + timedelta(minutes=1)
        )
        self.assertEqual(repair.question.objective_id, target_objective)
        self.assertNotEqual(
            repair.question.family_id, trigger_question.family_id
        )
        self.assertIn(wrong.misconception_id, repair.question.misconception_ids)

        repaired = self.engine.submit_answer(
            repair.decision_id,
            repair.question.correct_option.id,
            confidence=0.9,
            response_ms=1200,
            idempotency_key="objective-repair",
            now=answered_at + timedelta(minutes=2),
        )
        self.assertEqual(repaired.next_phase, SessionPhase.VERIFY)
        self.assertEqual(repaired.focus_objective_id, target_objective)

        verification = self.engine.next_question(
            session["id"], now=answered_at + timedelta(minutes=3)
        )
        self.assertEqual(verification.question.objective_id, target_objective)
        self.assertIn(
            wrong.misconception_id, verification.question.misconception_ids
        )
        self.assertNotIn(
            verification.question.family_id,
            {trigger_question.family_id, repair.question.family_id},
        )

    def test_cross_objective_diagnosis_rechecks_the_measured_parent(self) -> None:
        learner_id = "cross-objective-parent-obligation"
        self.engine.create_learner(learner_id)
        session = self.engine.start_session(
            learner_id,
            "t_transformers",
            mode="learn",
            seed=2,
            now=START,
        )
        trigger = self.engine.next_question(session["id"], now=START)
        self.assertEqual(
            trigger.question.id, "q_transformer_mask_direction_001"
        )
        self.assertEqual(
            trigger.question.objective_id, "lo_causal_visibility"
        )
        wrong = next(
            option for option in trigger.question.options if option.id == "b"
        )
        self.assertEqual(
            wrong.diagnostic_objective_id,
            "lo_transformer_information_paths",
        )
        self.assertEqual(
            wrong.misconception_id,
            "m_feedforward_layers_mix_token_positions",
        )

        diagnosed = self.engine.submit_answer(
            trigger.decision_id,
            wrong.id,
            confidence=0.9,
            response_ms=1200,
            idempotency_key="cross-objective-trigger",
            now=START + timedelta(minutes=1),
        )
        self.assertEqual(
            diagnosed.transition_reason,
            "cross_objective_diagnostic_focus",
        )
        self.assertEqual(diagnosed.next_phase, SessionPhase.REMEDIATE)
        self.assertEqual(
            diagnosed.focus_objective_id,
            "lo_transformer_information_paths",
        )
        self.assertEqual(
            diagnosed.focus_misconception_id,
            "m_feedforward_layers_mix_token_positions",
        )
        self.assertEqual(
            self.database.get_session(session["id"])["remediation_path"],
            [
                {
                    "concept_id": "c_causal_masking",
                    "objective_id": "lo_causal_visibility",
                    "misconception_id": None,
                }
            ],
        )

        child_repair = self.engine.next_question(
            session["id"], now=START + timedelta(minutes=2)
        )
        self.assertEqual(
            child_repair.question.objective_id,
            "lo_transformer_information_paths",
        )
        self.assertIn(
            wrong.misconception_id,
            child_repair.question.misconception_ids,
        )
        repaired = self.engine.submit_answer(
            child_repair.decision_id,
            child_repair.question.correct_option.id,
            confidence=0.9,
            response_ms=1200,
            idempotency_key="cross-objective-repair",
            now=START + timedelta(minutes=3),
        )
        self.assertEqual(
            repaired.transition_reason,
            "focused_repair_requires_independent_verification",
        )
        self.assertEqual(repaired.next_phase, SessionPhase.VERIFY)
        self.assertEqual(
            repaired.focus_objective_id,
            "lo_transformer_information_paths",
        )

        child_verification = self.engine.next_question(
            session["id"], now=START + timedelta(minutes=4)
        )
        self.assertEqual(
            child_verification.question.objective_id,
            "lo_transformer_information_paths",
        )
        self.assertIn(
            wrong.misconception_id,
            child_verification.question.misconception_ids,
        )
        self.assertNotEqual(
            child_verification.question.family_id,
            child_repair.question.family_id,
        )
        resumed = self.engine.submit_answer(
            child_verification.decision_id,
            child_verification.question.correct_option.id,
            confidence=0.9,
            response_ms=1200,
            idempotency_key="cross-objective-child-verification",
            now=START + timedelta(minutes=5),
        )
        self.assertEqual(
            resumed.transition_reason,
            "prerequisite_verified_resume_parent",
        )
        self.assertEqual(resumed.next_phase, SessionPhase.VERIFY)
        self.assertEqual(resumed.focus_objective_id, "lo_causal_visibility")
        self.assertEqual(resumed.focus_concept_id, "c_causal_masking")
        self.assertIsNone(resumed.focus_misconception_id)
        self.assertEqual(
            self.database.get_session(session["id"])["remediation_path"], []
        )

        parent_verification = self.engine.next_question(
            session["id"], now=START + timedelta(minutes=6)
        )
        self.assertEqual(
            parent_verification.question.objective_id,
            "lo_causal_visibility",
        )
        self.assertEqual(
            parent_verification.pedagogical_role, "verification"
        )
        families = {
            presentation.question.family_id
            for presentation in (
                trigger,
                child_repair,
                child_verification,
                parent_verification,
            )
        }
        self.assertEqual(len(families), 4)
        completed = self.engine.submit_answer(
            parent_verification.decision_id,
            parent_verification.question.correct_option.id,
            confidence=0.9,
            response_ms=1200,
            idempotency_key="cross-objective-parent-verification",
            now=START + timedelta(minutes=7),
        )
        self.assertEqual(
            completed.transition_reason,
            "independent_verification_completed",
        )
        self.assertEqual(completed.next_phase, SessionPhase.LEARN)

        report = self.engine.session_report(
            session["id"], now=START + timedelta(minutes=7)
        )
        routing = report["adaptive_routing"]
        self.assertEqual(routing["parent_resumptions"], 1)
        self.assertEqual(routing["prerequisite_resumptions"], 0)
        self.assertEqual(routing["cross_objective_parent_resumptions"], 1)
        self.assertEqual(routing["unclassified_parent_resumptions"], 0)
        resumed_step = next(
            step
            for step in report["adaptive_path"]
            if step["transition_reason"]
            == "prerequisite_verified_resume_parent"
        )
        self.assertEqual(
            resumed_step["parent_resume_origin"],
            "cross_objective_diagnostic",
        )

        trace = list(reversed(self.engine.trace(session["id"])))
        self.assertEqual(
            [row["question_id"] for row in trace],
            [
                trigger.question.id,
                child_repair.question.id,
                child_verification.question.id,
                parent_verification.question.id,
            ],
        )
        self.assertEqual(
            [row["phase"] for row in trace],
            ["learn", "remediate", "verify", "verify"],
        )
        self.assertEqual(
            [row["focus_objective_id"] for row in trace],
            [
                None,
                "lo_transformer_information_paths",
                "lo_transformer_information_paths",
                "lo_causal_visibility",
            ],
        )
        self.assertEqual(
            [row["focus_misconception_id"] for row in trace],
            [
                None,
                "m_feedforward_layers_mix_token_positions",
                "m_feedforward_layers_mix_token_positions",
                None,
            ],
        )
        integrity = self.database.verify_integrity()
        self.assertTrue(integrity["ok"], integrity["errors"])
        replay = ProjectionReplay(self.database).check(learner_id)
        self.assertTrue(replay["ok"], replay["errors"])
        self.assertTrue(replay["source_projection_matches_replay"])

    def test_cross_objective_route_preserves_an_only_parent_family(self) -> None:
        bundle = declared_fixture_bundle(
            load_bundle(CORPUS)
        )
        parent_objective = "lo_causal_visibility"
        child_objective = "lo_transformer_information_paths"
        shared_family = "f_adversarial_parent_verification"
        verification_kinds = {
            question_kind.value for question_kind in VERIFICATION_KINDS
        }
        for question in bundle["questions"]:
            if (
                question.get("learning_objective_id") == parent_objective
                and question["id"] != "q_transformer_mask_direction_001"
            ):
                if (
                    question["id"]
                    == "q_causal_prefix_extension_invariance_001"
                ):
                    # Keep one distinct, misconception-aligned repair family so
                    # the trigger remains pairwise serviceable.
                    question["kind"] = QuestionKind.CONCEPTUAL.value
                elif question["kind"] in verification_kinds:
                    # Every remaining parent verification aliases one family.
                    question["family_id"] = shared_family
            if (
                question["id"]
                == "q_transformer_cross_attention_direction_001"
            ):
                # This otherwise attractive child probe aliases the parent's
                # only transfer family and must never be consumed.
                question["family_id"] = shared_family

        parsed = parse_bundle(bundle)
        domains, topics = parse_catalog(bundle, parsed[0], parsed[4])
        database = Database(Path(self.tempdir.name) / "overlap.db")
        database.initialize()
        database.import_corpus(*parsed, domains, topics)
        engine = AdaptiveEngine(database)
        learner_id = "cross-objective-overlap"
        engine.create_learner(learner_id)
        session = engine.start_session(
            learner_id,
            "t_transformers",
            mode="learn",
            seed=2,
            now=START,
        )

        trigger = engine.next_question(session["id"], now=START)
        self.assertEqual(
            trigger.question.id, "q_transformer_mask_direction_001"
        )
        wrong = next(
            option for option in trigger.question.options if option.id == "b"
        )
        diagnosed = engine.submit_answer(
            trigger.decision_id,
            wrong.id,
            confidence=0.9,
            response_ms=1200,
            idempotency_key="overlap-trigger",
            now=START + timedelta(minutes=1),
        )
        self.assertEqual(diagnosed.focus_objective_id, child_objective)

        child_repair = engine.next_question(
            session["id"], now=START + timedelta(minutes=2)
        )
        self.assertEqual(child_repair.question.objective_id, child_objective)
        self.assertNotEqual(child_repair.question.family_id, shared_family)
        repair_trace = engine.trace(session["id"])[0]
        self.assertNotIn(
            "q_transformer_cross_attention_direction_001",
            {
                candidate["question_id"]
                for candidate in repair_trace["top_candidates"]
            },
        )
        repaired = engine.submit_answer(
            child_repair.decision_id,
            child_repair.question.correct_option.id,
            confidence=0.9,
            response_ms=1200,
            idempotency_key="overlap-repair",
            now=START + timedelta(minutes=3),
        )
        self.assertEqual(repaired.next_phase, SessionPhase.VERIFY)

        child_verification = engine.next_question(
            session["id"], now=START + timedelta(minutes=4)
        )
        self.assertEqual(
            child_verification.question.objective_id, child_objective
        )
        self.assertNotEqual(
            child_verification.question.family_id, shared_family
        )
        self.assertNotEqual(
            child_verification.question.family_id,
            child_repair.question.family_id,
        )
        resumed = engine.submit_answer(
            child_verification.decision_id,
            child_verification.question.correct_option.id,
            confidence=0.9,
            response_ms=1200,
            idempotency_key="overlap-child-verification",
            now=START + timedelta(minutes=5),
        )
        self.assertEqual(
            resumed.transition_reason,
            "prerequisite_verified_resume_parent",
        )
        self.assertEqual(resumed.focus_objective_id, parent_objective)

        parent_verification = engine.next_question(
            session["id"], now=START + timedelta(minutes=6)
        )
        self.assertEqual(
            parent_verification.question.objective_id, parent_objective
        )
        self.assertEqual(
            parent_verification.question.family_id, shared_family
        )
        self.assertEqual(
            len(
                {
                    trigger.question.family_id,
                    child_repair.question.family_id,
                    child_verification.question.family_id,
                    parent_verification.question.family_id,
                }
            ),
            4,
        )
        completed = engine.submit_answer(
            parent_verification.decision_id,
            parent_verification.question.correct_option.id,
            confidence=0.9,
            response_ms=1200,
            idempotency_key="overlap-parent-verification",
            now=START + timedelta(minutes=7),
        )
        self.assertEqual(completed.next_phase, SessionPhase.LEARN)
        integrity = database.verify_integrity()
        self.assertTrue(integrity["ok"], integrity["errors"])
        replay = ProjectionReplay(database).check(learner_id)
        self.assertTrue(replay["ok"], replay["errors"])

    def test_cross_objective_trigger_fails_closed_without_four_families(self) -> None:
        by_id = {question.id: question for question in self.questions}
        trigger = by_id["q_transformer_mask_direction_001"]
        trigger = replace(
            trigger,
            options=tuple(
                (
                    option
                    if option.id in {"b", "d"}
                    else replace(
                        option,
                        misconception_id=None,
                        diagnostic_objective_id=None,
                    )
                )
                for option in trigger.options
            ),
        )
        parent_repair = replace(
            by_id["q_causal_mask_training_leak_001"],
            family_id="f_synthetic_parent_repair",
        )
        parent_verification = replace(
            by_id["q_causal_mask_parallelism_001"],
            family_id="f_synthetic_shared",
        )
        child_repair = replace(
            by_id["q_transformer_cross_attention_direction_001"],
            family_id="f_synthetic_shared",
        )
        child_verification = replace(
            by_id["q_transformer_cross_attention_ablation_001"],
            family_id="f_synthetic_child_verification",
        )
        child_objective = "lo_transformer_information_paths"
        parent_objective = "lo_causal_visibility"

        # Pairwise reserves exist: parent repair -> shared verification, and
        # shared child repair -> child verification. The shared family makes
        # the complete trigger/repair/child-verify/parent-verify sequence
        # impossible, which is the case the older pairwise proof admitted.
        self.assertNotEqual(
            parent_repair.family_id, parent_verification.family_id
        )
        self.assertNotEqual(
            child_repair.family_id, child_verification.family_id
        )
        self.assertEqual(
            parent_verification.family_id, child_repair.family_id
        )

        def constrained_pool(
            _scope,
            *,
            focus_objective_id=None,
            **_kwargs,
        ):
            if focus_objective_id is None:
                return [trigger]
            if focus_objective_id == parent_objective:
                return [trigger, parent_repair, parent_verification]
            if focus_objective_id == child_objective:
                return [child_repair, child_verification]
            return []

        learner_id = "cross-objective-unserviceable"
        self.engine.create_learner(learner_id)
        session = self.engine.start_session(
            learner_id,
            "t_transformers",
            mode="learn",
            seed=2,
            now=START,
        )
        with patch.object(
            self.database,
            "questions_for_scope",
            side_effect=constrained_pool,
        ), self.assertRaisesRegex(
            ExhaustedError, "no safely serviceable main question"
        ):
            self.engine.next_question(session["id"], now=START)

    def test_objective_response_updates_no_concept_projection(self) -> None:
        learner_id = "direct-objective-only"
        self.engine.create_learner(learner_id)
        question = next(
            question
            for question in self.questions
            if question.objective_id is not None
            and any(
                mapping.role is not ConceptRole.PRIMARY
                and mapping.role.carries_scored_evidence
                for mapping in question.concepts
            )
        )
        with self.database.transaction() as connection:
            event = self.database.append_event(
                connection,
                stream_id=f"learner:{learner_id}",
                event_type="ObjectiveEvidenceTested",
                payload={"question_id": question.id},
                learner_id=learner_id,
                occurred_at=START,
            )
            concept_states, changes = self.engine.learner_model.update_from_response(
                connection,
                learner_id=learner_id,
                question=question,
                selected_option=question.correct_option,
                confidence=0.9,
                hint_count=0,
                feedback_shown=False,
                evidence_weight_override=1.0,
                event_id=event["event_id"],
                now=START,
                response_ms=1200,
                prior_family_attempts_override=0,
            )

        self.assertEqual(concept_states, {})
        self.assertEqual(self.database.get_skill_states(learner_id), {})
        self.assertEqual(
            [change.get("objective_id") for change in changes],
            [question.objective_id],
        )
        self.assertFalse(any("concept_id" in change for change in changes))
        self.assertEqual(
            set(self.database.get_objective_states(learner_id)),
            {question.objective_id},
        )

    def test_focused_repair_requires_diagnostic_objective_alignment(self) -> None:
        learner_id, session, trigger, wrong, target_objective = (
            self._serviceable_objective_trigger()
        )
        answered_at = START + timedelta(minutes=2)
        diagnosed = self.engine.submit_answer(
            trigger.decision_id,
            wrong.id,
            confidence=0.9,
            response_ms=1200,
            idempotency_key="diagnostic-alignment-trigger",
            now=answered_at,
        )
        self.assertEqual(diagnosed.focus_objective_id, target_objective)

        with self.database.read() as connection:
            owner = connection.execute(
                "SELECT concept_id FROM misconceptions WHERE id = ?",
                (wrong.misconception_id,),
            ).fetchone()["concept_id"]
        replacement = next(
            objective.id
            for objective in self.database.get_learning_objectives(
                self.release_id
            )
            if objective.id != target_objective
            and owner in objective.concept_ids
        )
        with self.database.transaction() as connection:
            aligned_candidates = connection.execute(
                """SELECT mapping.question_id, mapping.option_id
                   FROM release_option_objectives mapping
                   JOIN release_question_objectives direct
                     ON direct.release_id = mapping.release_id
                    AND direct.question_id = mapping.question_id
                   JOIN options option
                     ON option.question_id = mapping.question_id
                    AND option.option_id = mapping.option_id
                   WHERE mapping.release_id = ?
                     AND direct.objective_id = ?
                     AND mapping.objective_id = ?
                     AND option.misconception_id = ?
                     AND mapping.question_id != ?""",
                (
                    self.release_id,
                    target_objective,
                    target_objective,
                    wrong.misconception_id,
                    trigger.question.id,
                ),
            ).fetchall()
            self.assertTrue(aligned_candidates)
            self.database._drop_release_snapshot_triggers(connection)
            connection.executemany(
                """UPDATE release_option_objectives SET objective_id = ?
                   WHERE release_id = ? AND question_id = ? AND option_id = ?""",
                (
                    (
                        replacement,
                        self.release_id,
                        row["question_id"],
                        row["option_id"],
                    )
                    for row in aligned_candidates
                ),
            )

        with self.assertRaisesRegex(
            ExhaustedError, "no remediation-plus-verification pair"
        ):
            self.engine.next_question(
                session["id"], now=answered_at + timedelta(minutes=1)
            )

    def test_verified_prerequisite_not_reopened_after_parent_recheck_failure(
        self,
    ) -> None:
        learner_id = "objective-prerequisite-descent"
        self.engine.create_learner(learner_id)
        session = self.engine.start_session(
            learner_id, "t_transformers", mode="learn", seed=0
        )
        clock = START
        parent_frame = None
        ancestor_path: list[dict] = []
        descent = None

        # Seed zero currently reaches the cross-primary causal diagnosis first.
        # Keep driving credible main successes until that reviewed boundary is
        # presented so corpus additions cannot turn this into a question-ID test.
        for step in range(24):
            main = self.engine.next_question(session["id"], now=clock)
            clock += timedelta(minutes=1)
            causal_wrong = next(
                (
                    option
                    for option in main.question.options
                    if not option.correct
                    and option.diagnostic_objective_id
                    == "lo_causal_visibility"
                ),
                None,
            )
            if causal_wrong is None:
                self.engine.submit_answer(
                    main.decision_id,
                    main.question.correct_option.id,
                    confidence=0.9,
                    response_ms=1200,
                    idempotency_key=f"descent-skip-{step}",
                    now=clock,
                )
                clock += timedelta(minutes=1)
                continue

            diagnosed = self.engine.submit_answer(
                main.decision_id,
                causal_wrong.id,
                confidence=0.9,
                response_ms=1200,
                idempotency_key="descent-parent-diagnosis",
                now=clock,
            )
            clock += timedelta(minutes=1)
            parent_frame = {
                "concept_id": diagnosed.focus_concept_id,
                "objective_id": diagnosed.focus_objective_id,
                "misconception_id": diagnosed.focus_misconception_id,
            }
            ancestor_path = [
                dict(frame)
                for frame in self.database.get_session(session["id"])[
                    "remediation_path"
                ]
            ]
            self.assertEqual(diagnosed.next_phase, SessionPhase.REMEDIATE)

            focused = self.engine.next_question(session["id"], now=clock)
            clock += timedelta(minutes=1)
            focused_decision = self.database.recent_decisions(
                session["id"], 1
            )[0]
            self.assertEqual(
                focused_decision["focus_objective_id"],
                parent_frame["objective_id"],
            )
            self.assertTrue(focused_decision["focus_valid"])
            second_wrong = next(
                option
                for option in focused.question.options
                if not option.correct
                and option.misconception_id
                == parent_frame["misconception_id"]
                and (
                    option.diagnostic_objective_id
                    or focused.question.objective_id
                )
                == parent_frame["objective_id"]
            )
            descent = self.engine.submit_answer(
                focused.decision_id,
                second_wrong.id,
                confidence=0.9,
                response_ms=1200,
                idempotency_key="descent-second-focused-wrong",
                now=clock,
            )
            clock += timedelta(minutes=1)
            if descent.transition_reason == "descend_to_evidence_boundary":
                break
            parent_frame = None
            ancestor_path = []
            descent = None

        self.assertIsNotNone(descent, "seed-zero trace never reached a descent")
        self.assertIsNotNone(parent_frame)
        self.assertIsNotNone(descent.focus_objective_id)
        objectives = {
            objective.id: objective
            for objective in self.database.get_learning_objectives(
                self.release_id
            )
        }
        child_objective = objectives[descent.focus_objective_id]
        self.assertEqual(
            descent.focus_concept_id,
            child_objective.primary_concept_id,
        )
        parent_objective = objectives[parent_frame["objective_id"]]
        declared_direct = {
            edge.source_id for edge in parent_objective.prerequisites
        }
        self.assertIn(descent.focus_objective_id, declared_direct)
        self.assertIsNotNone(descent.boundary_decision)
        self.assertEqual(
            descent.boundary_decision["focus_objective_id"],
            parent_frame["objective_id"],
        )
        self.assertEqual(
            descent.boundary_decision["selected_objective_id"],
            descent.focus_objective_id,
        )
        self.assertEqual(
            descent.boundary_decision["selected"]["objective_id"],
            descent.focus_objective_id,
        )
        self.assertEqual(
            self.database.get_session(session["id"])["remediation_path"],
            [*ancestor_path, parent_frame],
        )

        child_repair = self.engine.next_question(session["id"], now=clock)
        clock += timedelta(minutes=1)
        self.assertEqual(
            child_repair.question.objective_id, descent.focus_objective_id
        )
        repaired = self.engine.submit_answer(
            child_repair.decision_id,
            child_repair.question.correct_option.id,
            confidence=0.9,
            response_ms=1200,
            idempotency_key="descent-child-repair",
            now=clock,
        )
        clock += timedelta(minutes=1)
        self.assertEqual(repaired.next_phase, SessionPhase.VERIFY)
        self.assertEqual(
            repaired.focus_objective_id, descent.focus_objective_id
        )

        child_verification = self.engine.next_question(
            session["id"], now=clock
        )
        clock += timedelta(minutes=1)
        self.assertEqual(
            child_verification.question.objective_id,
            descent.focus_objective_id,
        )
        self.assertNotEqual(
            child_verification.question.family_id,
            child_repair.question.family_id,
        )
        resumed = self.engine.submit_answer(
            child_verification.decision_id,
            child_verification.question.correct_option.id,
            confidence=0.9,
            response_ms=1200,
            idempotency_key="descent-child-verification",
            now=clock,
        )
        clock += timedelta(minutes=1)

        self.assertEqual(
            resumed.transition_reason,
            "prerequisite_verified_resume_parent",
        )
        self.assertEqual(resumed.next_phase, SessionPhase.VERIFY)
        self.assertEqual(resumed.focus_concept_id, parent_frame["concept_id"])
        self.assertEqual(
            resumed.focus_objective_id, parent_frame["objective_id"]
        )
        self.assertEqual(
            resumed.focus_misconception_id,
            parent_frame["misconception_id"],
        )
        self.assertEqual(
            self.database.get_session(session["id"])["remediation_path"],
            ancestor_path,
        )
        with self.database.read() as connection:
            session_certificates = (
                self.engine._same_session_verified_objective_focuses(
                    connection,
                    session_id=session["id"],
                    release_id=self.release_id,
                    objective_ids={descent.focus_objective_id},
                )
            )
        self.assertIn(
            (descent.focus_objective_id, None),
            session_certificates,
        )

        parent_recheck = self.engine.next_question(
            session["id"], now=clock
        )
        clock += timedelta(minutes=1)
        self.assertEqual(
            parent_recheck.question.objective_id,
            parent_frame["objective_id"],
        )
        self.assertEqual(
            parent_recheck.pedagogical_role,
            "verification",
        )
        parent_wrong = next(
            option
            for option in parent_recheck.question.options
            if not option.correct
        )
        not_reopened = self.engine.submit_answer(
            parent_recheck.decision_id,
            parent_wrong.id,
            confidence=0.9,
            response_ms=1200,
            idempotency_key="descent-parent-recheck-wrong",
            now=clock,
        )

        self.assertEqual(
            not_reopened.transition_reason,
            "verified_prerequisite_not_reopened",
        )
        self.assertEqual(not_reopened.next_phase, SessionPhase.LEARN)
        self.assertIsNone(not_reopened.boundary_decision)
        self.assertEqual(
            self.database.get_session(session["id"])["remediation_path"],
            [],
        )
        trace = self.engine.session_report(
            session["id"], now=clock
        )["adaptive_path"]
        self.assertEqual(
            [
                row["transition_reason"]
                for row in trace[-4:]
            ],
            [
                "descend_to_evidence_boundary",
                "focused_repair_requires_independent_verification",
                "prerequisite_verified_resume_parent",
                "verified_prerequisite_not_reopened",
            ],
        )
        replay = ProjectionReplay(self.database).check(learner_id)
        self.assertTrue(replay["ok"], replay["errors"])
        self.assertTrue(replay["source_projection_matches_replay"])

    def test_focused_objective_retrieval_bypasses_broad_scope_cutoff(self) -> None:
        objective_by_id = {
            objective.id: objective
            for objective in self.database.get_learning_objectives(
                self.release_id
            )
        }
        cross_primary = next(
            question
            for question in self.questions
            if question.objective_id is not None
            and question.primary_concept_id
            != objective_by_id[
                question.objective_id
            ].primary_concept_id
        )
        objective = objective_by_id[cross_primary.objective_id]
        decoy_count = 650
        imported_at = START.isoformat()
        question_rows = []
        concept_rows = []
        release_rows = []
        for index in range(decoy_count):
            question_id = f"q_objective_cutoff_decoy_{index:04d}"
            question_rows.append(
                (
                    question_id,
                    1,
                    f"{index:064x}"[-64:],
                    f"f_objective_cutoff_decoy_{index:04d}",
                    QuestionStatus.APPROVED.value,
                    "Which unrelated decoy occupies this broad concept slot?",
                    QuestionKind.CONCEPTUAL.value,
                    0.0,
                    2.0,
                    0.25,
                    0.05,
                    "{}",
                    "[]",
                    None,
                    imported_at,
                )
            )
            concept_rows.append(
                (
                    question_id,
                    objective.primary_concept_id,
                    1.0,
                    ConceptRole.PRIMARY.value,
                )
            )
            release_rows.append(
                (
                    self.release_id,
                    question_id,
                    QuestionStatus.APPROVED.value,
                    QuestionStatus.APPROVED.evidence_weight,
                )
            )

        with self.database.transaction() as connection:
            self.database._drop_release_snapshot_triggers(connection)
            connection.executemany(
                """INSERT INTO questions(
                       id, version, content_hash, family_id, status, stem, kind,
                       difficulty, discrimination, guess_rate, slip_rate,
                       provenance_json, tags_json, revision_of, imported_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                question_rows,
            )
            connection.executemany(
                """INSERT INTO question_concepts(
                       question_id, concept_id, weight, role
                   ) VALUES (?, ?, ?, ?)""",
                concept_rows,
            )
            connection.executemany(
                """INSERT INTO release_questions(
                       release_id, question_id, status, evidence_weight
                   ) VALUES (?, ?, ?, ?)""",
                release_rows,
            )

        broad = self.database.questions_for_scope(
            {objective.primary_concept_id},
            focus_concept_id=objective.primary_concept_id,
            release_id=self.release_id,
            target_difficulty=0.0,
            limit=600,
        )
        focused = self.database.questions_for_scope(
            {objective.primary_concept_id},
            focus_concept_id=objective.primary_concept_id,
            focus_objective_id=objective.id,
            release_id=self.release_id,
            target_difficulty=0.0,
            limit=600,
        )

        self.assertEqual(len(broad), 600)
        self.assertNotIn(cross_primary.id, {question.id for question in broad})
        self.assertIn(cross_primary.id, {question.id for question in focused})
        self.assertTrue(
            all(question.objective_id == objective.id for question in focused)
        )

    def test_objective_history_keeps_later_concept_projection_on_v3_hash(self) -> None:
        learner_id = "objective-then-concept"
        self.engine.create_learner(learner_id)
        objective_session = self.engine.start_session(
            learner_id, "t_transformers", mode="learn", seed=0
        )
        objective_question = self.engine.next_question(
            objective_session["id"], now=START
        )
        self.assertIsNotNone(objective_question.question.objective_id)
        self.engine.submit_answer(
            objective_question.decision_id,
            objective_question.question.correct_option.id,
            confidence=0.9,
            response_ms=1200,
            idempotency_key="objective-history-first",
            now=START + timedelta(minutes=1),
        )

        concept_session = self.engine.start_session(
            learner_id, "c_clustering", mode="learn", seed=0
        )
        concept_question = self.engine.next_question(
            concept_session["id"], now=START + timedelta(days=1)
        )
        self.assertIsNone(concept_question.question.objective_id)
        self.engine.submit_answer(
            concept_question.decision_id,
            concept_question.question.correct_option.id,
            confidence=0.9,
            response_ms=1200,
            idempotency_key="objective-history-second",
            now=START + timedelta(days=1, minutes=1),
        )

        with self.database.read() as connection:
            projections = connection.execute(
                """SELECT schema_version, payload_json FROM events
                   WHERE learner_id = ?
                     AND event_type = 'LearnerProjectionAdvanced'
                   ORDER BY stream_version""",
                (learner_id,),
            ).fetchall()
        self.assertEqual(
            [row["schema_version"] for row in projections], [4, 4]
        )
        latest_payload = json.loads(projections[-1]["payload_json"])
        self.assertEqual(latest_payload["projection_hash_version"], 3)
        self.assertIsNone(latest_payload["question_objective_id"])
        self.assertEqual(
            latest_payload["projection_hash"],
            self.database.learner_projection_hash(learner_id, hash_version=3),
        )
        integrity = self.database.verify_integrity()
        self.assertTrue(integrity["ok"], integrity["errors"])
        replay = ProjectionReplay(self.database).check(learner_id)
        self.assertTrue(replay["ok"], replay["errors"])
        self.assertTrue(replay["source_projection_matches_replay"])
        self.assertTrue(replay["commitment_matches_replay"])

    def test_pre_objective_model_cannot_select_or_apply_objective_evidence(self) -> None:
        learner_id = "pre-objective-model"
        self.engine.create_learner(learner_id)
        session = self.engine.start_session(
            learner_id, "t_transformers", mode="learn", seed=0
        )
        legacy_engine = AdaptiveEngine(
            self.database, LearnerModel(CONCEPT_MODEL_VERSION)
        )

        with self.assertRaisesRegex(
            ValidationError, "cannot select objective-aware questions"
        ):
            legacy_engine.next_question(session["id"], now=START)

        with self.database.read() as connection:
            after_rejected_selection = {
                "decisions": connection.execute(
                    "SELECT COUNT(*) AS n FROM decisions WHERE learner_id = ?",
                    (learner_id,),
                ).fetchone()["n"],
                "attempts": connection.execute(
                    "SELECT COUNT(*) AS n FROM attempts WHERE learner_id = ?",
                    (learner_id,),
                ).fetchone()["n"],
                "skill_states": connection.execute(
                    "SELECT COUNT(*) AS n FROM skill_states WHERE learner_id = ?",
                    (learner_id,),
                ).fetchone()["n"],
                "objective_states": connection.execute(
                    "SELECT COUNT(*) AS n FROM objective_states WHERE learner_id = ?",
                    (learner_id,),
                ).fetchone()["n"],
                "selection_events": connection.execute(
                    """SELECT COUNT(*) AS n FROM events
                       WHERE learner_id = ? AND event_type = 'QuestionSelected'""",
                    (learner_id,),
                ).fetchone()["n"],
                "learner_revision": connection.execute(
                    "SELECT revision FROM learners WHERE id = ?",
                    (learner_id,),
                ).fetchone()["revision"],
                "session_step": connection.execute(
                    "SELECT step FROM sessions WHERE id = ?",
                    (session["id"],),
                ).fetchone()["step"],
            }
        self.assertEqual(
            after_rejected_selection,
            {
                "decisions": 0,
                "attempts": 0,
                "skill_states": 0,
                "objective_states": 0,
                "selection_events": 0,
                "learner_revision": 0,
                "session_step": 0,
            },
        )

        presentation = self.engine.next_question(session["id"], now=START)
        self.assertIsNotNone(presentation.question.objective_id)
        with self.assertRaisesRegex(
            ConflictError, "stale decision was invalidated"
        ):
            legacy_engine.submit_answer(
                presentation.decision_id,
                presentation.question.correct_option.id,
                confidence=0.9,
                response_ms=1200,
                now=START + timedelta(minutes=1),
            )

        with self.database.read() as connection:
            pending = connection.execute(
                "SELECT consumed_at, invalidation_reason FROM decisions WHERE id = ?",
                (presentation.decision_id,),
            ).fetchone()
            attempt_count = connection.execute(
                "SELECT COUNT(*) AS n FROM attempts WHERE learner_id = ?",
                (learner_id,),
            ).fetchone()["n"]
            learner_revision = connection.execute(
                "SELECT revision FROM learners WHERE id = ?",
                (learner_id,),
            ).fetchone()["revision"]
        self.assertIsNone(pending["consumed_at"])
        self.assertEqual(pending["invalidation_reason"], "learner_model_changed")
        self.assertEqual(attempt_count, 0)
        self.assertEqual(learner_revision, 0)
        self.assertEqual(self.database.get_skill_states(learner_id), {})
        self.assertEqual(self.database.get_objective_states(learner_id), {})

        replacement = self.engine.next_question(
            session["id"], now=START + timedelta(minutes=1)
        )
        applied = self.engine.submit_answer(
            replacement.decision_id,
            replacement.question.correct_option.id,
            confidence=0.9,
            response_ms=1200,
            idempotency_key="current-objective-model",
            now=START + timedelta(minutes=1, milliseconds=1200),
        )
        self.assertTrue(applied.correct)
        self.assertIn(
            replacement.question.objective_id,
            self.database.get_objective_states(learner_id),
        )
        integrity = self.database.verify_integrity()
        self.assertTrue(integrity["ok"], integrity["errors"])

    def test_bound_objective_selection_anchors_action_replay(self) -> None:
        learner_id = "objective-selection-action"
        self.engine.create_learner(learner_id)
        session = self.engine.start_session(
            learner_id, "t_transformers", mode="learn", seed=0
        )
        presentation = self.engine.next_question(session["id"], now=START)
        self.assertIsNotNone(presentation.question.objective_id)

        with self.database.read() as connection:
            selections = connection.execute(
                """SELECT schema_version, payload_json FROM events
                   WHERE learner_id = ? AND session_id = ?
                     AND event_type = 'QuestionSelected'
                   ORDER BY stream_version""",
                (learner_id, session["id"]),
            ).fetchall()
        self.assertEqual(len(selections), 1)
        self.assertEqual(
            selections[0]["schema_version"],
            BOUND_QUESTION_SELECTED_EVENT_SCHEMA_VERSION,
        )
        selection_payload = json.loads(selections[0]["payload_json"])
        self.assertEqual(
            selection_payload["question_objective_id"],
            presentation.question.objective_id,
        )
        self.assertIsNone(selection_payload["focus_objective_id"])

        action = self.engine.record_action(
            presentation.decision_id,
            "started",
            {},
            idempotency_key="objective-selection-action",
            now=START + timedelta(seconds=1),
        )
        self.assertEqual(action["decision_id"], presentation.decision_id)
        integrity = self.database.verify_integrity()
        self.assertTrue(integrity["ok"], integrity["errors"])
        replay = ProjectionReplay(self.database).check(learner_id)
        self.assertTrue(replay["ok"], replay["errors"])
        self.assertEqual(replay["action_event_count"], 1)
        self.assertTrue(replay["action_projection_matches_replay"])

    def test_action_and_integrity_reject_stale_session_objective_focus(self) -> None:
        learner_id = "stale-objective-action"
        self.engine.create_learner(learner_id)
        session = self.engine.start_session(
            learner_id, "t_transformers", mode="learn", seed=0
        )
        presentation = self.engine.next_question(session["id"], now=START)
        replacement = next(
            objective.id
            for objective in self.database.get_learning_objectives(
                self.release_id
            )
            if objective.id != presentation.question.objective_id
        )
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE sessions SET focus_objective_id = ? WHERE id = ?",
                (replacement, session["id"]),
            )

        rejected = False
        try:
            self.engine.record_action(
                presentation.decision_id,
                "started",
                {},
                idempotency_key="stale-objective-action",
                now=START + timedelta(seconds=1),
            )
        except ConflictError:
            rejected = True
        with self.database.read() as connection:
            action_count = connection.execute(
                "SELECT COUNT(*) AS n FROM learning_actions WHERE learner_id = ?",
                (learner_id,),
            ).fetchone()["n"]
        integrity = self.database.verify_integrity()

        with self.subTest(boundary="record_action"):
            self.assertTrue(
                rejected,
                "record_action accepted a decision whose objective focus was stale",
            )
            self.assertEqual(action_count, 0)
        with self.subTest(boundary="integrity"):
            self.assertFalse(
                integrity["ok"],
                "integrity accepted a session/decision objective-focus mismatch",
            )
            self.assertTrue(
                any(
                    presentation.decision_id in error
                    and "focus objective" in error.casefold()
                    for error in integrity["errors"]
                ),
                integrity["errors"],
            )

    def test_session_report_exposes_objective_scoped_evidence(self) -> None:
        learner_id = "objective-session-report"
        self.engine.create_learner(learner_id)
        session = self.engine.start_session(
            learner_id, "t_transformers", mode="learn", seed=0
        )
        presentation = self.engine.next_question(session["id"], now=START)
        result = self.engine.submit_answer(
            presentation.decision_id,
            presentation.question.correct_option.id,
            confidence=0.9,
            response_ms=1200,
            idempotency_key="objective-session-report",
            now=START + timedelta(minutes=1),
        )
        report_now = START + timedelta(minutes=2)
        report = self.engine.session_report(session["id"], now=report_now)
        objective_id = presentation.question.objective_id
        objective_definition = presentation.question.objective
        stored = self.database.get_objective_states(learner_id)[objective_id]
        projected = self.engine.learner_model.project_objective_state(
            stored, objective_definition, report_now
        )
        expected_change = next(
            change
            for change in result.state_changes
            if change.get("objective_id") == objective_id
        )
        with self.database.read() as connection:
            family_count = connection.execute(
                """SELECT COUNT(*) AS n FROM learner_objective_families
                   WHERE learner_id = ? AND objective_id = ?""",
                (learner_id, objective_id),
            ).fetchone()["n"]

        with self.subTest(section="objective_changes"):
            self.assertIn("objective_changes", report)
            changes = {
                change["objective_id"]: change
                for change in report["objective_changes"]
            }
            self.assertEqual(set(changes), {objective_id})
            self.assertAlmostEqual(
                changes[objective_id]["prior_mastery"],
                expected_change["prior_mastery"],
            )
            self.assertAlmostEqual(
                changes[objective_id]["posterior_mastery"],
                expected_change["posterior_mastery"],
            )
            self.assertAlmostEqual(
                changes[objective_id]["evidence_delta"],
                expected_change["evidence_delta"],
            )
        with self.subTest(section="objective_performance"):
            self.assertIn("objective_performance", report)
            performance = {
                item["objective_id"]: item
                for item in report["objective_performance"]
            }
            self.assertEqual(set(performance), {objective_id})
            objective_report = performance[objective_id]
            self.assertEqual(objective_report["name"], objective_definition.name)
            self.assertEqual(
                objective_report["operation"],
                objective_definition.operation.value,
            )
            self.assertEqual(
                objective_report["evidence_type"],
                objective_definition.evidence_type,
            )
            self.assertEqual(objective_report["session"]["attempted"], 1)
            self.assertEqual(objective_report["session"]["correct"], 1)
            current = objective_report["current_projection"]
            self.assertAlmostEqual(
                current["mastery_probability"], projected.mastery_probability
            )
            self.assertAlmostEqual(
                current["expected_competence"], projected.expected_competence
            )
            self.assertAlmostEqual(
                current["uncertainty"], projected.variance**0.5
            )
            self.assertAlmostEqual(current["evidence_mass"], projected.evidence_mass)
            self.assertEqual(current["independent_families"], family_count)

    def test_sealed_objective_mapping_tamper_is_detected(self) -> None:
        clean = self.database.verify_integrity()
        self.assertTrue(clean["ok"], clean["errors"])
        with self.database.read() as connection:
            mapping = connection.execute(
                """SELECT question_id, objective_id
                   FROM release_question_objectives
                   WHERE release_id = ? ORDER BY question_id LIMIT 1""",
                (self.release_id,),
            ).fetchone()
            replacement = connection.execute(
                """SELECT objective_id FROM release_learning_objectives
                   WHERE release_id = ? AND objective_id != ?
                   ORDER BY objective_id LIMIT 1""",
                (self.release_id, mapping["objective_id"]),
            ).fetchone()["objective_id"]

        with self.assertRaises(sqlite3.IntegrityError):
            with self.database.transaction() as connection:
                connection.execute(
                    """UPDATE release_question_objectives SET objective_id = ?
                       WHERE release_id = ? AND question_id = ?""",
                    (replacement, self.release_id, mapping["question_id"]),
                )

        with self.database.transaction() as connection:
            self.database._drop_release_snapshot_triggers(connection)
            connection.execute(
                """UPDATE release_question_objectives SET objective_id = ?
                   WHERE release_id = ? AND question_id = ?""",
                (replacement, self.release_id, mapping["question_id"]),
            )

        corrupted = self.database.verify_integrity()
        self.assertFalse(corrupted["ok"])
        self.assertIn(
            f"release {self.release_id}: bundle hash mismatch",
            corrupted["errors"],
        )

    def test_sealed_objective_graph_is_immutable_and_cycles_fail_closed(self) -> None:
        clean = self.database.verify_integrity()
        self.assertTrue(clean["ok"], clean["errors"])
        with self.assertRaises(sqlite3.IntegrityError):
            with self.database.transaction() as connection:
                connection.execute(
                    """UPDATE release_objective_edges SET weight = 0.5
                       WHERE release_id = ?
                         AND edge_id = 'oe_value_prereq_order'""",
                    (self.release_id,),
                )

        with self.database.transaction() as connection:
            self.database._drop_release_snapshot_triggers(connection)
            connection.execute(
                """UPDATE release_objective_edges
                   SET source_objective_id = 'lo_transformer_sublayer_composition',
                       target_objective_id = 'lo_attention_value_routing'
                   WHERE release_id = ?
                     AND edge_id = 'oe_value_prereq_order'""",
                (self.release_id,),
            )

        corrupted = self.database.verify_integrity()
        self.assertFalse(corrupted["ok"])
        self.assertIn(
            f"release {self.release_id}: objective prerequisite graph contains a cycle",
            corrupted["errors"],
        )
        self.assertIn(
            f"release {self.release_id}: bundle hash mismatch",
            corrupted["errors"],
        )
        with self.assertRaisesRegex(ValidationError, "contains a cycle"):
            self.database.get_objective_graph(self.release_id)

    def test_v4_concept_and_v5_objective_history_replay_together(self) -> None:
        mixed = Database(Path(self.tempdir.name) / "mixed.db")
        mixed.initialize()
        raw = declared_fixture_bundle(
            load_bundle(CORPUS)
        )
        v1 = legacy_bundle(raw)
        parsed_v1 = parse_bundle(v1)
        catalog_v1 = parse_catalog(v1, parsed_v1[0], parsed_v1[4])
        legacy_release = mixed.import_corpus(*parsed_v1, *catalog_v1)["release_id"]

        legacy_engine = AdaptiveEngine(
            mixed, LearnerModel(CONCEPT_MODEL_VERSION)
        )
        legacy_engine.create_learner("mixed-objective-history")
        legacy_session = legacy_engine.start_session(
            "mixed-objective-history", "c_clustering", seed=17
        )
        legacy_question = legacy_engine.next_question(
            legacy_session["id"], now=START
        )
        self.assertIsNone(legacy_question.question.objective_id)
        legacy_engine.submit_answer(
            legacy_question.decision_id,
            legacy_question.question.correct_option.id,
            confidence=0.9,
            response_ms=1200,
            now=START + timedelta(minutes=1),
        )

        parsed_current = parse_bundle(raw)
        catalog_current = parse_catalog(
            raw, parsed_current[0], parsed_current[4]
        )
        current_release = mixed.import_corpus(
            *parsed_current, *catalog_current
        )["release_id"]
        self.assertNotEqual(legacy_release, current_release)
        current_engine = AdaptiveEngine(mixed)
        objective_session = current_engine.start_session(
            "mixed-objective-history", "t_transformers", seed=0
        )
        objective_question = current_engine.next_question(
            objective_session["id"], now=START + timedelta(days=1)
        )
        self.assertIsNotNone(objective_question.question.objective_id)
        current_engine.submit_answer(
            objective_question.decision_id,
            objective_question.question.correct_option.id,
            confidence=0.9,
            response_ms=1200,
            now=START + timedelta(days=1, minutes=1),
        )

        report = ProjectionReplay(mixed).check("mixed-objective-history")

        self.assertTrue(report["ok"], report["errors"])
        self.assertTrue(report["source_projection_matches_replay"])
        self.assertTrue(report["commitment_matches_replay"])
        with mixed.read() as connection:
            checkpoints = connection.execute(
                """SELECT schema_version, metadata_json FROM events
                   WHERE learner_id = ?
                     AND event_type = 'LearnerProjectionAdvanced'
                   ORDER BY stream_version""",
                ("mixed-objective-history",),
            ).fetchall()
        self.assertEqual(
            [row["schema_version"] for row in checkpoints], [2, 4]
        )
        self.assertEqual(
            {
                json.loads(row["metadata_json"])["learner_model_version"]
                for row in checkpoints
            },
            {CONCEPT_MODEL_VERSION, MODEL_VERSION},
        )


if __name__ == "__main__":
    unittest.main()
