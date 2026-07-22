#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0

"""Exercise TSQ's real adaptive engine as an inspectable behavioral system.

This is deliberately outside ``tests/``.  It is a repeatable experimental
instrument: action policies supply answers, while all question selection,
remediation, verification, state updates, event persistence, and graph traversal
remain production TSQ behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import operator
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tsq.adaptive import BOUNDARY_ALGORITHM_VERSION  # noqa: E402
from tsq.corpus import read_and_parse  # noqa: E402
from tsq.engine import AdaptiveEngine  # noqa: E402
from tsq.errors import NotFoundError  # noqa: E402
from tsq.models import Presentation, SessionPhase  # noqa: E402
from tsq.policy import POLICY_VERSION  # noqa: E402
from tsq.simulation import (  # noqa: E402
    BehavioralSimulator,
    SimulationReport,
    SyntheticAnswer,
    SyntheticLearner,
)
from tsq.store import Database  # noqa: E402


LAB_VERSION = "adaptive-behavior-lab-v1"
DEFAULT_START = datetime(2100, 7, 21, 9, 0, tzinfo=timezone.utc)
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "results" / "adaptive_lab.json"
DEFAULT_ROOT = "t_transformers"


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    id: str
    behavior: str
    hypothesis: str


SCENARIOS = (
    ScenarioSpec(
        "deliberate_correct",
        "Every answer is correct, high-confidence, unhinted, and takes 8 seconds.",
        "Credible retrieval should accumulate independent evidence and improve projections.",
    ),
    ScenarioSpec(
        "fast_correct",
        "Every answer is correct and high-confidence but takes only 100 ms.",
        "Implausibly fast correctness should be discounted and should not certify families.",
    ),
    ScenarioSpec(
        "hinted_correct",
        "Every answer is correct and high-confidence after one hint.",
        "Hinted success should carry less independent evidence than unassisted success.",
    ),
    ScenarioSpec(
        "low_confidence_correct",
        "Every answer is correct and unhinted but confidence remains 0.20.",
        "Correctness paired with self-reported uncertainty should be discounted.",
    ),
    ScenarioSpec(
        "uncertain_abstention",
        "Every item is explicitly unanswered with confidence 0.20.",
        "Abstention should preserve uncertainty without inventing a named misconception.",
    ),
    ScenarioSpec(
        "confident_misconception",
        "Every answer selects a deterministic named misconception with confidence 0.95.",
        "Confident errors should open bounded repair and create misconception hypotheses.",
    ),
    ScenarioSpec(
        "fixed_option_bias",
        "The lexicographically first option is selected on every item.",
        "A repeated response-position habit should not resemble credible mastery.",
    ),
    ScenarioSpec(
        "repair_on_support",
        "Main-phase answers are wrong; remediation and verification answers are correct.",
        "Successful support should resolve focused episodes and return to the parent path.",
    ),
    ScenarioSpec(
        "verification_lapse",
        "Main and verification answers are wrong, while remediation answers are correct.",
        "A repair answer without independent verification should not produce stable mastery.",
    ),
    ScenarioSpec(
        "oscillating_answers",
        "Answers repeat a deterministic wrong, wrong, correct, correct cycle.",
        "Nonstationary evidence should remain visible as uncertainty and focused routing.",
    ),
    ScenarioSpec(
        "targeted_attention_gap",
        "Answers are correct except on causal masking and attention scaling.",
        "The learner graph should localize weakness instead of lowering every skill equally.",
    ),
    ScenarioSpec(
        "heterogeneous_profile",
        "A seeded probabilistic learner is strong generally but weak on masking and scaling.",
        "Repeated samples should expose the targeted weak boundary under realistic lapses.",
    ),
)
SCENARIO_BY_ID = {scenario.id: scenario for scenario in SCENARIOS}


@dataclass(slots=True)
class PatternLearner:
    """Deterministic action policy used only to provide answers to TSQ.

    The engine still controls which item is seen and how the observation changes
    learner state.  Incorrect responses always choose a corpus-authored named
    misconception when one is available.
    """

    name: str
    rule: str
    confidence: float
    response_ms: int
    hint_count: int = 0
    weak_concepts: frozenset[str] = field(default_factory=frozenset)
    calls: int = 0

    def answer(
        self,
        presentation: Presentation,
        *,
        simulation_seed: int,
        trial_index: int,
        encounter: int,
    ) -> SyntheticAnswer:
        del trial_index
        call_index = self.calls
        self.calls += 1
        if self.rule == "fixed_option":
            selected = min(presentation.question.options, key=lambda option: option.id)
            return SyntheticAnswer(
                selected_option_id=selected.id,
                correct=selected.correct,
                ground_truth_probability=0.25,
                confidence=self.confidence,
                response_ms=self.response_ms,
                hint_count=self.hint_count,
            )
        outcome = self._outcome(presentation, call_index)
        if outcome is None:
            return SyntheticAnswer(
                selected_option_id=None,
                correct=False,
                ground_truth_probability=0.50,
                confidence=self.confidence,
                response_ms=self.response_ms,
                hint_count=self.hint_count,
            )
        if outcome:
            selected = presentation.question.correct_option
        else:
            selected = self._named_distractor(
                presentation,
                seed=simulation_seed,
                encounter=encounter,
            )
        return SyntheticAnswer(
            selected_option_id=selected.id,
            correct=outcome,
            ground_truth_probability=0.98 if outcome else 0.02,
            confidence=self.confidence,
            response_ms=self.response_ms,
            hint_count=self.hint_count,
        )

    def _outcome(
        self, presentation: Presentation, call_index: int
    ) -> bool | None:
        if self.rule == "correct":
            return True
        if self.rule == "wrong":
            return False
        if self.rule == "abstain":
            return None
        if self.rule == "repair":
            return presentation.phase in {
                SessionPhase.REMEDIATE,
                SessionPhase.VERIFY,
            }
        if self.rule == "verification_lapse":
            return presentation.phase == SessionPhase.REMEDIATE
        if self.rule == "oscillate":
            return call_index % 4 in {2, 3}
        if self.rule == "targeted":
            return presentation.question.primary_concept_id not in self.weak_concepts
        raise ValueError(f"Unknown laboratory action rule: {self.rule}")

    def _named_distractor(
        self,
        presentation: Presentation,
        *,
        seed: int,
        encounter: int,
    ):
        question = presentation.question
        distractors = sorted(
            (option for option in question.options if not option.correct),
            key=lambda option: option.id,
        )
        named = [option for option in distractors if option.misconception_id]
        candidates = named or distractors
        if not candidates:
            raise RuntimeError(f"Question {question.id} has no distractor.")
        material = f"{self.name}|{seed}|{question.id}|{encounter}".encode()
        index = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        return candidates[index % len(candidates)]


def make_learner(scenario_id: str, seed: int):
    if scenario_id == "deliberate_correct":
        return PatternLearner(scenario_id, "correct", 0.95, 8_000)
    if scenario_id == "fast_correct":
        return PatternLearner(scenario_id, "correct", 0.95, 100)
    if scenario_id == "hinted_correct":
        return PatternLearner(scenario_id, "correct", 0.95, 8_000, hint_count=1)
    if scenario_id == "low_confidence_correct":
        return PatternLearner(scenario_id, "correct", 0.20, 8_000)
    if scenario_id == "uncertain_abstention":
        return PatternLearner(scenario_id, "abstain", 0.20, 5_000)
    if scenario_id == "confident_misconception":
        return PatternLearner(scenario_id, "wrong", 0.95, 3_500)
    if scenario_id == "fixed_option_bias":
        return PatternLearner(scenario_id, "fixed_option", 0.75, 1_200)
    if scenario_id == "repair_on_support":
        return PatternLearner(scenario_id, "repair", 0.82, 6_000)
    if scenario_id == "verification_lapse":
        return PatternLearner(scenario_id, "verification_lapse", 0.88, 5_500)
    if scenario_id == "oscillating_answers":
        return PatternLearner(scenario_id, "oscillate", 0.70, 4_500)
    if scenario_id == "targeted_attention_gap":
        return PatternLearner(
            scenario_id,
            "targeted",
            0.90,
            7_000,
            weak_concepts=frozenset(
                {"c_attention_scaling", "c_causal_masking"}
            ),
        )
    if scenario_id == "heterogeneous_profile":
        return SyntheticLearner(
            scenario_id,
            default_ability=0.88,
            concept_abilities={
                "c_attention": 0.86,
                "c_attention_scaling": 0.12,
                "c_causal_masking": 0.08,
                "c_transformers": 0.72,
            },
            misconception_strengths={
                "m_attention_unscaled_dimension_invariant": 0.90,
                "m_mask_only_inference": 0.90,
            },
            slip_probability=0.03,
            guess_probability=0.01,
            base_response_ms=6_000,
            seed=seed,
        )
    raise ValueError(f"Unknown scenario: {scenario_id}")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def corpus_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_profile_references(
    learner: PatternLearner | SyntheticLearner,
    database: Database,
    release_id: str,
) -> None:
    """Fail early when an experimental profile names nonexistent graph objects."""

    graph = database.get_graph(release_id)
    if isinstance(learner, PatternLearner):
        concept_ids = set(learner.weak_concepts)
        misconception_ids: set[str] = set()
    else:
        concept_ids = set(learner.concept_abilities)
        misconception_ids = set(learner.misconception_strengths)
    unknown_concepts = concept_ids - set(graph.concepts)
    if unknown_concepts:
        raise ValueError(
            "Laboratory profile references unknown concepts: "
            + ", ".join(sorted(unknown_concepts))
        )
    if misconception_ids:
        known_misconceptions = {
            item.id
            for item in database.get_misconceptions(
                misconception_ids, release_id=release_id
            )
        }
        unknown_misconceptions = misconception_ids - known_misconceptions
        if unknown_misconceptions:
            raise ValueError(
                "Laboratory profile references unknown misconceptions: "
                + ", ".join(sorted(unknown_misconceptions))
            )


def projection_summary(profile: Mapping[str, Any]) -> dict[str, Any]:
    skills = list(profile["skills"])
    observed = [skill for skill in skills if skill["evidence_mass"] > 0]
    misconceptions = list(profile["active_misconceptions"])
    by_concept = {
        skill["concept_id"]: {
            key: skill[key]
            for key in (
                "name",
                "mastery",
                "expected_competence",
                "uncertainty",
                "stability_hours",
                "evidence_mass",
                "independent_families",
                "delayed_retrievals",
                "operation_kinds",
                "prerequisites_ready",
                "intrinsic_readiness",
                "prerequisite_support",
                "effective_readiness",
                "bottleneck_concept_id",
                "state",
                "next_review_at",
            )
        }
        for skill in skills
    }
    stable_payload = {
        "skills": by_concept,
        "active_misconceptions": misconceptions,
    }
    return {
        "skill_count": len(skills),
        "observed_skill_count": len(observed),
        "total_evidence_mass": sum(skill["evidence_mass"] for skill in skills),
        "total_independent_families": sum(
            skill["independent_families"] for skill in skills
        ),
        "mean_mastery": (
            sum(skill["mastery"] for skill in skills) / len(skills)
            if skills
            else None
        ),
        "mean_observed_mastery": (
            sum(skill["mastery"] for skill in observed) / len(observed)
            if observed
            else None
        ),
        "state_counts": dict(Counter(skill["state"] for skill in skills)),
        "active_misconception_count": len(misconceptions),
        "maximum_misconception_probability": max(
            (item["probability"] for item in misconceptions),
            default=None,
        ),
        "skills": by_concept,
        "active_misconceptions": misconceptions,
        "stable_hash": canonical_hash(stable_payload),
    }


def graph_snapshot(
    database: Database,
    *,
    root_reference: str,
    release_id: str,
) -> dict[str, Any]:
    graph = database.get_graph(release_id)
    try:
        topic = database.resolve_topic(root_reference, release_id)
    except NotFoundError:  # The engine uses the same topic-or-concept resolution.
        topic = None
    if topic is not None:
        scope = database.topic_scope(topic["id"], release_id)
        target = {
            "type": "topic",
            "id": topic["id"],
            "name": topic["name"],
        }
    else:
        scope = graph.learning_scope(root_reference)
        target = {
            "type": "concept",
            "id": root_reference,
            "name": graph.concepts[root_reference].name,
        }
    nodes = []
    for concept_id in sorted(scope):
        concept = graph.concepts[concept_id]
        nodes.append(
            {
                "id": concept_id,
                "name": concept.name,
                "prior_mastery": concept.prior_mastery,
                "direct_prerequisites": [
                    {"concept_id": prerequisite_id, "weight": weight}
                    for prerequisite_id, weight in graph.direct_prerequisites(
                        concept_id
                    )
                ],
            }
        )
    edges = [
        {
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "relation": edge.relation.value,
            "weight": edge.weight,
        }
        for edge in graph.edges
        if edge.source_id in scope and edge.target_id in scope
    ]
    return {
        "target": target,
        "scope_concepts": len(scope),
        "nodes": nodes,
        "edges": edges,
    }


def _session_for_learner(database: Database, learner_id: str) -> dict[str, Any]:
    with database.read() as connection:
        row = connection.execute(
            """SELECT id FROM sessions WHERE learner_id = ?
               ORDER BY created_at DESC, id DESC LIMIT 1""",
            (learner_id,),
        ).fetchone()
    if row is None:
        raise RuntimeError(f"No session was created for {learner_id}.")
    return database.get_session(row["id"])


def _topic_names(database: Database, release_id: str) -> dict[str, str]:
    catalog = database.get_catalog(release_id)
    return {topic["id"]: topic["name"] for topic in catalog["topics"]}


def serialize_trace(
    report: SimulationReport,
    session_report: Mapping[str, Any],
    database: Database,
    session: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    graph = database.get_graph(session["corpus_release_id"])
    topic_names = _topic_names(database, session["corpus_release_id"])
    root_id = session["root_concept_id"]
    distances = graph.learning_distances_to(root_id) if root_id else {}
    adaptive_path = list(session_report["adaptive_path"])
    violations: list[str] = []
    if len(adaptive_path) != len(report.steps):
        violations.append(
            "simulation step count differs from persisted adaptive path count"
        )
    trace = []
    for index, step in enumerate(report.steps):
        persisted = adaptive_path[index] if index < len(adaptive_path) else {}
        focus_before = step.focus_concept_before
        focus_after = step.focus_concept_after
        trace.append(
            {
                "step": step.index + 1,
                "phase": {
                    "before": step.phase_before.value,
                    "after": step.phase_after.value,
                },
                "question": {
                    "id": step.question_id,
                    "family_id": step.family_id,
                    "kind": step.question_kind,
                    "pedagogical_role": step.pedagogical_role,
                    "primary_concept_id": step.primary_concept_id,
                    "primary_concept_name": graph.concepts[
                        step.primary_concept_id
                    ].name,
                    "learning_distance_to_root": distances.get(
                        step.primary_concept_id
                    ),
                    "topics": [
                        {"id": topic_id, "name": topic_names.get(topic_id)}
                        for topic_id in step.topic_ids
                    ],
                },
                "observation": {
                    "correct": step.actual_correct,
                    "selected_option_id": step.selected_option_id,
                    "selected_misconception_id": persisted.get(
                        "selected_misconception_id"
                    ),
                    "selected_misconception_name": persisted.get(
                        "selected_misconception_name"
                    ),
                    "confidence": step.confidence,
                    "response_ms": step.response_ms,
                    "hint_count": step.hint_count,
                    "ground_truth_probability": step.ground_truth_probability,
                },
                "selection": {
                    "predicted_correct": step.predicted_correct,
                    "continuity": step.continuity,
                    "exact_repeat": step.exact_repeat,
                    "family_repeat": step.family_repeat,
                },
                "focus": {
                    "before": focus_before,
                    "before_name": (
                        graph.concepts[focus_before].name if focus_before else None
                    ),
                    "after": focus_after,
                    "after_name": (
                        graph.concepts[focus_after].name if focus_after else None
                    ),
                },
                "routing": {
                    "transition_reason": persisted.get("transition_reason"),
                    "boundary_decision": persisted.get("boundary_decision"),
                },
                "clock": {
                    "selected_at": step.selected_at.isoformat(),
                    "answered_at": step.answered_at.isoformat(),
                },
            }
        )
    return trace, violations


def serialize_episodes(
    report: SimulationReport, database: Database, release_id: str
) -> list[dict[str, Any]]:
    graph = database.get_graph(release_id)
    return [
        {
            "start_step": episode.start_step + 1,
            "end_step": episode.end_step + 1,
            "trigger_question_id": episode.trigger_question_id,
            "trigger_family_id": episode.trigger_family_id,
            "initial_focus_concept_id": episode.initial_focus_concept_id,
            "initial_focus_concept_name": (
                graph.concepts[episode.initial_focus_concept_id].name
                if episode.initial_focus_concept_id
                else None
            ),
            "initial_focus_misconception_id": (
                episode.initial_focus_misconception_id
            ),
            "focus_path": [
                {"concept_id": concept_id, "name": graph.concepts[concept_id].name}
                for concept_id in episode.focus_path
            ],
            "question_ids": list(episode.question_ids),
            "family_ids": list(episode.family_ids),
            "length": episode.length,
            "outcome": episode.outcome,
            "exact_repeat_count": episode.exact_repeat_count,
            "family_repeat_count": episode.family_repeat_count,
        }
        for episode in report.focus_episodes
    ]


def compact_session_report(report: Mapping[str, Any]) -> dict[str, Any]:
    retained = (
        "questions_presented",
        "questions_answered",
        "correct",
        "accuracy",
        "abstained",
        "unique_families",
        "unique_concepts",
        "phase_counts",
        "difficulty",
        "average_predicted_success",
        "continuity",
        "exploration",
        "remediation_questions",
        "cross_topic_questions",
        "topic_distribution",
        "evidence_delta",
        "concept_changes",
        "adaptive_routing",
        "concept_performance",
        "diagnostic_findings",
        "diagnostic_contract",
    )
    return {key: report[key] for key in retained}


def capacity_and_demand_snapshot(
    database: Database, session: Mapping[str, Any]
) -> dict[str, Any]:
    """Expose unused family capacity and authoring work created by a live gap."""

    release_id = session["corpus_release_id"]
    if session.get("topic_id"):
        owned = database.topic_owned_concepts(
            session["topic_id"], release_id, include_descendants=True
        )
    else:
        owned = {session["root_concept_id"]}
    placeholders = ",".join("?" for _ in owned)
    with database.read() as connection:
        rows = connection.execute(
            f"""SELECT mapping.concept_id, question.id AS question_id,
                       question.family_id, question.kind
                FROM release_questions membership
                JOIN questions question ON question.id = membership.question_id
                JOIN question_concepts mapping
                  ON mapping.question_id = question.id
                 AND mapping.role = 'primary'
                WHERE membership.release_id = ?
                  AND membership.status IN ('approved', 'calibrated')
                  AND mapping.concept_id IN ({placeholders})
                ORDER BY mapping.concept_id, question.family_id, question.id""",
            (release_id, *sorted(owned)),
        ).fetchall()
        used_families = {
            row["family_id"]
            for row in connection.execute(
                """SELECT DISTINCT question.family_id
                   FROM decisions decision
                   JOIN questions question ON question.id = decision.question_id
                   WHERE decision.session_id = ?""",
                (session["id"],),
            )
        }
        jobs = [
            {
                "status": row["status"],
                "provider": row["provider"],
                "model": row["model"],
                "prompt_version": row["prompt_version"],
                "blueprint": json.loads(row["blueprint_json"]),
            }
            for row in connection.execute(
                """SELECT blueprint_json, status, provider, model, prompt_version
                   FROM generation_jobs ORDER BY blueprint_json"""
            )
        ]

    verification_kinds = {
        "application",
        "calculation",
        "comparison",
        "counterfactual",
        "debugging",
        "transfer",
    }
    by_concept: dict[str, dict[str, set[str] | int]] = {
        concept_id: {
            "questions": 0,
            "families": set(),
            "verification_families": set(),
        }
        for concept_id in owned
    }
    for row in rows:
        values = by_concept[row["concept_id"]]
        values["questions"] = int(values["questions"]) + 1
        families = values["families"]
        assert isinstance(families, set)
        families.add(row["family_id"])
        if row["kind"] in verification_kinds:
            verification = values["verification_families"]
            assert isinstance(verification, set)
            verification.add(row["family_id"])

    graph = database.get_graph(release_id)
    concepts = []
    for concept_id in sorted(owned):
        values = by_concept[concept_id]
        families = values["families"]
        verification = values["verification_families"]
        assert isinstance(families, set)
        assert isinstance(verification, set)
        used = families & used_families
        concepts.append(
            {
                "concept_id": concept_id,
                "name": graph.concepts[concept_id].name,
                "approved_questions": values["questions"],
                "independent_families": len(families),
                "verification_families": len(verification),
                "used_families": len(used),
                "remaining_families": len(families - used_families),
                "remaining_verification_families": len(
                    verification - used_families
                ),
            }
        )
    return {
        "owned_concepts": concepts,
        "used_session_families": len(used_families),
        "remaining_owned_families": sum(
            concept["remaining_families"] for concept in concepts
        ),
        "generation_demands_created": jobs,
    }


def scenario_invariants(
    report: SimulationReport,
    trace_violations: Iterable[str],
    integrity: Mapping[str, Any],
) -> list[str]:
    failures = list(trace_violations)
    if not integrity["ok"]:
        failures.append(
            "database integrity failed: " + "; ".join(integrity["errors"][:5])
        )
    if report.remediation_exact_repeat_count:
        failures.append("an exact question repeated inside a focused episode")
    if report.remediation_family_repeat_count:
        failures.append("an item family repeated inside a focused episode")
    for episode in report.focus_episodes:
        if episode.trigger_question_id in episode.question_ids:
            failures.append(
                f"episode at step {episode.start_step} reused its trigger question"
            )
        if episode.trigger_family_id in episode.family_ids:
            failures.append(
                f"episode at step {episode.start_step} reused its trigger family"
            )
    return failures


def run_scenario(
    spec: ScenarioSpec,
    *,
    database_path: Path,
    corpus_path: Path,
    root_reference: str,
    steps: int,
    seed: int,
    replay: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    database = Database(database_path)
    database.initialize()
    release = database.import_corpus(
        *read_and_parse(corpus_path, include_catalog=True)
    )
    engine = AdaptiveEngine(database)
    simulator = BehavioralSimulator(engine)
    learner_id = f"lab-{spec.id}"
    learner = make_learner(spec.id, seed)
    validate_profile_references(learner, database, release["release_id"])
    report = simulator.run(
        learner,
        learner_id=learner_id,
        root_concept_id=root_reference,
        policy_seed=seed,
        max_steps=steps,
        start_at=DEFAULT_START,
    )
    session = _session_for_learner(database, learner_id)
    persisted_report = engine.session_report(session["id"], now=report.ended_at)
    profile = projection_summary(
        engine.profile(learner_id, root_concept_id=root_reference, now=report.ended_at)
    )
    trace, trace_violations = serialize_trace(
        report, persisted_report, database, session
    )
    integrity = database.verify_integrity()
    invariant_failures = scenario_invariants(
        report, trace_violations, integrity
    )
    replay_result: dict[str, Any] | None = None
    if replay:
        replay_path = database_path.with_name(database_path.stem + "-replay.db")
        replay_database = Database(replay_path)
        replay_database.initialize()
        replay_database.import_corpus(
            *read_and_parse(corpus_path, include_catalog=True)
        )
        replay_engine = AdaptiveEngine(replay_database)
        replay_report = BehavioralSimulator(replay_engine).run(
            make_learner(spec.id, seed),
            learner_id=learner_id,
            root_concept_id=root_reference,
            policy_seed=seed,
            max_steps=steps,
            start_at=DEFAULT_START,
        )
        replay_profile = projection_summary(
            replay_engine.profile(
                learner_id,
                root_concept_id=root_reference,
                now=replay_report.ended_at,
            )
        )
        replay_integrity = replay_database.verify_integrity()
        signature_matches = (
            report.behavior_signature() == replay_report.behavior_signature()
        )
        projection_matches = profile["stable_hash"] == replay_profile["stable_hash"]
        if not signature_matches:
            invariant_failures.append("fresh-database behavior replay diverged")
        if not projection_matches:
            invariant_failures.append("fresh-database learner projection diverged")
        if not replay_integrity["ok"]:
            invariant_failures.append("replay database integrity failed")
        replay_result = {
            "checked": True,
            "behavior_signature_matches": signature_matches,
            "projection_matches": projection_matches,
            "database_integrity_ok": replay_integrity["ok"],
        }

    summary = report.summary()
    result = {
        "id": spec.id,
        "behavior": spec.behavior,
        "hypothesis": spec.hypothesis,
        "behavior_signature": report.behavior_signature(),
        "summary": summary,
        "projection": profile,
        "session": compact_session_report(persisted_report),
        "capacity_after_run": capacity_and_demand_snapshot(database, session),
        "trace": trace,
        "remediation_episodes": serialize_episodes(
            report, database, session["corpus_release_id"]
        ),
        "gaps": summary["blockers"],
        "integrity": {
            "ok": integrity["ok"],
            "event_count": integrity["event_count"],
            "stream_count": integrity["stream_count"],
            "errors": integrity["errors"],
        },
        "deterministic_replay": replay_result or {"checked": False},
        "invariant_failures": invariant_failures,
    }
    return result, graph_snapshot(
        database,
        root_reference=root_reference,
        release_id=release["release_id"],
    )


@dataclass(frozen=True, slots=True)
class ComparisonSpec:
    id: str
    left: str
    right: str
    metric: str
    relation: str
    statement: str


COMPARISONS = (
    ComparisonSpec(
        "credible_over_fast_evidence",
        "deliberate_correct",
        "fast_correct",
        "projection.total_evidence_mass",
        ">",
        "Deliberate correct work earns more evidence than 100 ms correctness.",
    ),
    ComparisonSpec(
        "credible_over_fast_families",
        "deliberate_correct",
        "fast_correct",
        "projection.total_independent_families",
        ">",
        "Deliberate correct work certifies more independent families.",
    ),
    ComparisonSpec(
        "credible_over_hinted_evidence",
        "deliberate_correct",
        "hinted_correct",
        "projection.total_evidence_mass",
        ">",
        "Unhinted correct work earns more evidence than hinted success.",
    ),
    ComparisonSpec(
        "credible_over_uncertain_correct_evidence",
        "deliberate_correct",
        "low_confidence_correct",
        "projection.total_evidence_mass",
        ">",
        "High-confidence credible work earns more evidence than uncertain correctness.",
    ),
    ComparisonSpec(
        "credible_over_fixed_option_mastery",
        "deliberate_correct",
        "fixed_option_bias",
        "projection.mean_mastery",
        ">",
        "Credible work produces higher mastery than a fixed-option response habit.",
    ),
    ComparisonSpec(
        "credible_over_wrong_mastery",
        "deliberate_correct",
        "confident_misconception",
        "projection.mean_mastery",
        ">",
        "Consistently credible correctness produces a higher mean projection.",
    ),
    ComparisonSpec(
        "wrong_over_abstain_misconceptions",
        "confident_misconception",
        "uncertain_abstention",
        "projection.active_misconception_count",
        ">",
        "Named wrong choices create more misconception hypotheses than abstention.",
    ),
    ComparisonSpec(
        "verification_lapse_detected",
        "verification_lapse",
        "repair_on_support",
        "session.adaptive_routing.bounded_exits",
        ">=",
        "Verification lapses cause at least as many bounded focused exits as recovery.",
    ),
    ComparisonSpec(
        "targeted_masking_localization",
        "deliberate_correct",
        "targeted_attention_gap",
        "projection.skills.c_causal_masking.mastery",
        ">",
        "Targeted causal-masking errors lower that skill relative to credible work.",
    ),
)


RELATIONS: dict[str, Callable[[Any, Any], bool]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
}


def dotted_value(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for component in path.split("."):
        current = current[component]
    return current


def compare_scenarios(
    scenario_results: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    comparisons = []
    for spec in COMPARISONS:
        if spec.left not in scenario_results or spec.right not in scenario_results:
            continue
        try:
            left_value = dotted_value(scenario_results[spec.left], spec.metric)
            right_value = dotted_value(scenario_results[spec.right], spec.metric)
        except KeyError as error:
            comparisons.append(
                {
                    "id": spec.id,
                    "statement": spec.statement,
                    "left_scenario": spec.left,
                    "right_scenario": spec.right,
                    "metric": spec.metric,
                    "expected_relation": spec.relation,
                    "status": "not_applicable",
                    "reason": f"metric is outside this run's graph: {error}",
                }
            )
            continue
        supported = RELATIONS[spec.relation](left_value, right_value)
        comparisons.append(
            {
                "id": spec.id,
                "statement": spec.statement,
                "left_scenario": spec.left,
                "right_scenario": spec.right,
                "metric": spec.metric,
                "expected_relation": spec.relation,
                "left_value": left_value,
                "right_value": right_value,
                "difference": (
                    left_value - right_value
                    if isinstance(left_value, (int, float))
                    and isinstance(right_value, (int, float))
                    else None
                ),
                "status": "supported" if supported else "contradicted",
            }
        )
    return comparisons


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Run deterministic adversarial learners through the real TSQ engine "
            "and persist an inspectable JSON artifact."
        )
    )
    result.add_argument(
        "--corpus",
        type=Path,
        default=PROJECT_ROOT / "corpus" / "ai_curriculum.json",
    )
    result.add_argument("--root", default=DEFAULT_ROOT)
    result.add_argument("--steps", type=int, default=24)
    result.add_argument("--seed", type=int, default=71)
    result.add_argument(
        "--scenario",
        action="append",
        choices=tuple(SCENARIO_BY_ID),
        help="run only this scenario; repeat to select multiple scenarios",
    )
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument(
        "--database-dir",
        type=Path,
        help="preserve synthetic SQLite databases in this directory",
    )
    result.add_argument(
        "--no-replay-check",
        action="store_true",
        help="skip the second fresh-database determinism run",
    )
    result.add_argument(
        "--fail-on-hypothesis",
        action="store_true",
        help="exit 3 when a cross-profile behavioral hypothesis is contradicted",
    )
    result.add_argument(
        "--stdout",
        action="store_true",
        help="print the complete JSON artifact instead of a compact receipt",
    )
    result.add_argument(
        "--list-scenarios",
        action="store_true",
        help="list scenario IDs and exit",
    )
    return result


def execute(arguments: argparse.Namespace, database_directory: Path) -> dict[str, Any]:
    selected_ids = arguments.scenario or [scenario.id for scenario in SCENARIOS]
    results = []
    topology: dict[str, Any] | None = None
    for scenario_id in selected_ids:
        spec = SCENARIO_BY_ID[scenario_id]
        scenario_result, scenario_topology = run_scenario(
            spec,
            database_path=database_directory / f"{scenario_id}.db",
            corpus_path=arguments.corpus,
            root_reference=arguments.root,
            steps=arguments.steps,
            seed=arguments.seed,
            replay=not arguments.no_replay_check,
        )
        results.append(scenario_result)
        topology = topology or scenario_topology

    by_id = {result["id"]: result for result in results}
    comparisons = compare_scenarios(by_id)
    invariant_failures = [
        {"scenario": result["id"], "failure": failure}
        for result in results
        for failure in result["invariant_failures"]
    ]
    contradictions = [
        comparison
        for comparison in comparisons
        if comparison["status"] == "contradicted"
    ]
    checked_comparisons = [
        comparison
        for comparison in comparisons
        if comparison["status"] != "not_applicable"
    ]
    blockers = [
        {"scenario": result["id"], **gap}
        for result in results
        for gap in result["gaps"]
    ]
    return {
        "lab_version": LAB_VERSION,
        "engine_versions": {
            "policy": POLICY_VERSION,
            "boundary_algorithm": BOUNDARY_ALGORITHM_VERSION,
        },
        "configuration": {
            "corpus": str(arguments.corpus.resolve()),
            "corpus_sha256": corpus_sha256(arguments.corpus),
            "root": arguments.root,
            "steps_per_scenario": arguments.steps,
            "policy_seed": arguments.seed,
            "simulation_start": DEFAULT_START.isoformat(),
            "fresh_database_replay": not arguments.no_replay_check,
            "scenario_ids": selected_ids,
        },
        "observation_boundary": {
            "currently_observed": [
                "presented question and option set",
                "final selected option or explicit abstention",
                "self-reported confidence",
                "response latency",
                "hint count",
                "session phase and adaptive focus",
            ],
            "semantic_ledger_only": [
                "digest-only answer and artifact checkpoints",
                "allowlisted check-result summaries",
                "hint, tool-purpose, submission, and feedback actions",
            ],
            "not_yet_scored_online": [
                "free-form explanation or reasoning trace",
                "raw code edits and reversals",
                "raw compiler, runtime, and test output",
                "artifact quality and rubric-level outcomes",
                "longitudinal real-human transfer outcomes",
            ],
            "interpretation": (
                "The laboratory measures the online adaptive projection only. "
                "Semantic actions remain observational, and neither final-answer "
                "signals nor unscored traces are a complete model of human skill."
            ),
        },
        "curriculum_graph": topology,
        "scenarios": results,
        "comparisons": comparisons,
        "audit": {
            "hard_invariants_ok": not invariant_failures,
            "invariant_failures": invariant_failures,
            "behavioral_hypotheses_supported": len(checked_comparisons)
            - len(contradictions),
            "behavioral_hypotheses_checked": len(checked_comparisons),
            "behavioral_hypotheses_not_applicable": len(comparisons)
            - len(checked_comparisons),
            "contradicted_hypotheses": contradictions,
            "corpus_or_exhaustion_gaps": blockers,
            "scenario_count": len(results),
            "total_answers": sum(
                result["summary"]["attempted"] for result in results
            ),
            "all_integrity_checks_ok": all(
                result["integrity"]["ok"] for result in results
            ),
            "all_replays_deterministic": all(
                not result["deterministic_replay"].get("checked")
                or (
                    result["deterministic_replay"][
                        "behavior_signature_matches"
                    ]
                    and result["deterministic_replay"]["projection_matches"]
                )
                for result in results
            ),
        },
    }


def main() -> int:
    arguments = parser().parse_args()
    if arguments.list_scenarios:
        for scenario in SCENARIOS:
            print(f"{scenario.id}: {scenario.behavior}")
        return 0
    if arguments.steps <= 0:
        raise SystemExit("--steps must be positive")
    if not arguments.corpus.is_file():
        raise SystemExit(f"Corpus does not exist: {arguments.corpus}")

    if arguments.database_dir:
        arguments.database_dir.mkdir(parents=True, exist_ok=True)
        run_directory = Path(
            tempfile.mkdtemp(prefix="run-", dir=arguments.database_dir)
        )
        artifact = execute(arguments, run_directory)
        artifact["configuration"]["preserved_database_directory"] = str(
            run_directory.resolve()
        )
    else:
        with tempfile.TemporaryDirectory(prefix="tsq-adaptive-lab-") as directory:
            artifact = execute(arguments, Path(directory))

    write_json_atomic(arguments.output, artifact)
    if arguments.stdout:
        print(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False))
    else:
        receipt = {
            "output": str(arguments.output.resolve()),
            "audit": artifact["audit"],
            "comparisons": artifact["comparisons"],
        }
        print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))

    if not artifact["audit"]["hard_invariants_ok"]:
        return 2
    if (
        arguments.fail_on_hypothesis
        and artifact["audit"]["contradicted_hypotheses"]
    ):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
