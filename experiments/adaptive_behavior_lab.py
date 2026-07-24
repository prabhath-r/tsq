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
from datetime import datetime, timedelta, timezone
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
from tsq.objective_posterior import decode_objective_posterior  # noqa: E402
from tsq.policy import POLICY_VERSION  # noqa: E402
from tsq.replay import ProjectionReplay  # noqa: E402
from tsq.simulation import (  # noqa: E402
    BehavioralSimulator,
    SIMULATION_FEEDBACK_PROTOCOL_VERSION,
    SimulationReport,
    SyntheticAnswer,
    SyntheticLearner,
)
from tsq.store import Database  # noqa: E402


LAB_VERSION = "adaptive-behavior-lab-v7"
SEMANTIC_PROJECTION_SIGNATURE_SCHEMA = 1
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
        "missing_confidence_correct",
        "Every answer is correct and unhinted, but confidence telemetry is omitted.",
        "Missing confidence may update uncertainty but must never certify retrieval.",
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
        "The first displayed option is selected on every item.",
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
    confidence: float | None
    response_ms: int
    hint_count: int = 0
    weak_concepts: frozenset[str] = field(default_factory=frozenset)
    weak_objectives: frozenset[str] = field(default_factory=frozenset)
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
            selected = presentation.ordered_options[0]
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
            question = presentation.question
            if question.objective_id is not None:
                return question.objective_id not in self.weak_objectives
            anchor = (
                question.objective.primary_concept_id
                if question.objective is not None
                else question.primary_concept_id
            )
            return anchor not in self.weak_concepts
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


@dataclass(frozen=True, slots=True)
class MisconceptionProbeLearner:
    """Induce one named hypothesis, or answer correctly to test recovery."""

    name: str
    target_misconception_id: str
    induce: bool

    def answer(
        self,
        presentation: Presentation,
        *,
        simulation_seed: int,
        trial_index: int,
        encounter: int,
    ) -> SyntheticAnswer:
        del simulation_seed, trial_index, encounter
        target_options = [
            option
            for option in presentation.question.options
            if option.misconception_id == self.target_misconception_id
        ]
        selected = (
            target_options[0]
            if self.induce and target_options
            else presentation.question.correct_option
        )
        return SyntheticAnswer(
            selected_option_id=selected.id,
            correct=selected.correct,
            ground_truth_probability=0.02 if not selected.correct else 0.98,
            confidence=0.95,
            response_ms=4_000,
            hint_count=0,
        )


def make_learner(scenario_id: str, seed: int):
    if scenario_id == "deliberate_correct":
        return PatternLearner(scenario_id, "correct", 0.95, 8_000)
    if scenario_id == "fast_correct":
        return PatternLearner(scenario_id, "correct", 0.95, 100)
    if scenario_id == "hinted_correct":
        return PatternLearner(scenario_id, "correct", 0.95, 8_000, hint_count=1)
    if scenario_id == "low_confidence_correct":
        return PatternLearner(scenario_id, "correct", 0.20, 8_000)
    if scenario_id == "missing_confidence_correct":
        return PatternLearner(scenario_id, "correct", None, 8_000)
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
            weak_objectives=frozenset(
                {"lo_attention_logit_scaling", "lo_causal_visibility"}
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
            objective_abilities={
                "lo_attention_logit_scaling": 0.12,
                "lo_causal_visibility": 0.08,
            },
            misconception_strengths={
                "m_attention_unscaled_dimension_invariant": 0.90,
                "m_mask_only_inference": 0.90,
            },
            slip_probability=0.03,
            guess_probability=0.01,
            base_response_ms=6_000,
            response_model="discontinuous_threshold",
            seed=seed,
        )
    raise ValueError(f"Unknown scenario: {scenario_id}")


def generator_summary(learner: PatternLearner | SyntheticLearner) -> dict[str, Any]:
    if isinstance(learner, PatternLearner):
        return {
            "model": f"pattern:{learner.rule}",
            "weak_concepts": sorted(learner.weak_concepts),
            "weak_objectives": sorted(learner.weak_objectives),
            "forced_pattern": True,
        }
    return {
        "model": learner.response_model,
        "default_ability": learner.default_ability,
        "default_objective_ability": learner.default_objective_ability,
        "concept_abilities": dict(learner.concept_abilities),
        "objective_abilities": dict(learner.objective_abilities),
        "misconception_strengths": dict(learner.misconception_strengths),
        "forced_correctness": learner.forced_correctness,
        "interpretation": (
            "A declared synthetic response generator, potentially misspecified "
            "relative to TSQ's learner model; not a human calibration cohort."
        ),
    }


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
        objective_ids = set(learner.weak_objectives)
        misconception_ids: set[str] = set()
    else:
        concept_ids = set(learner.concept_abilities)
        objective_ids = set(learner.objective_abilities)
        misconception_ids = set(learner.misconception_strengths)
    unknown_concepts = concept_ids - set(graph.concepts)
    if unknown_concepts:
        raise ValueError(
            "Laboratory profile references unknown concepts: "
            + ", ".join(sorted(unknown_concepts))
        )
    known_objectives = {
        objective.id for objective in database.get_learning_objectives(release_id)
    }
    unknown_objectives = objective_ids - known_objectives
    if unknown_objectives:
        raise ValueError(
            "Laboratory profile references unknown learning objectives: "
            + ", ".join(sorted(unknown_objectives))
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


def _projection_commitment(database: Database, learner_id: str) -> dict[str, Any]:
    """Bind a readable lab summary to the database's complete projection hash."""

    with database.read() as connection:
        row = connection.execute(
            """SELECT payload_json FROM events
               WHERE learner_id = ? AND event_type = 'LearnerProjectionAdvanced'
               ORDER BY stream_version DESC LIMIT 1""",
            (learner_id,),
        ).fetchone()
        payload = json.loads(row["payload_json"]) if row is not None else None
        hash_version = (
            int(payload.get("projection_hash_version", 1))
            if isinstance(payload, dict)
            else None
        )
        current_hash = database.learner_projection_hash(
            learner_id,
            connection,
            **(
                {"hash_version": hash_version}
                if hash_version is not None
                else {}
            ),
        )
    committed_hash = (
        payload.get("projection_hash") if isinstance(payload, dict) else None
    )
    return {
        "hash_version": hash_version,
        "current_hash": current_hash,
        "latest_event_hash": committed_hash,
        "matches_latest_event": (
            committed_hash == current_hash if committed_hash is not None else None
        ),
    }


def semantic_projection_signature(
    database: Database, learner_id: str
) -> dict[str, Any]:
    """Commit to learner semantics without binding fresh-run event identities.

    Production projection commitments intentionally bind ``as_of_event_id`` and
    exact posterior observation IDs.  That is the right contract when replaying
    one immutable event history, but independent deterministic simulations
    create different event IDs.  This laboratory signature retains every
    numerical state, posterior density, evidence-family certificate, timestamp,
    model identity, and misconception belief while removing only those event
    provenance identifiers.  The exact production commitment remains checked
    separately.
    """

    with database.read() as connection:
        learner = connection.execute(
            "SELECT id, revision FROM learners WHERE id = ?",
            (learner_id,),
        ).fetchone()
        if learner is None:
            raise NotFoundError(f"Unknown learner: {learner_id}")

        skill_states = [
            dict(row)
            for row in connection.execute(
                """SELECT learner_id, concept_id, mean, variance,
                          stability_hours, exposures, last_seen_at,
                          next_review_at, evidence_mass, model_version
                   FROM skill_states WHERE learner_id = ?
                   ORDER BY concept_id""",
                (learner_id,),
            )
        ]
        objective_states = [
            dict(row)
            for row in connection.execute(
                """SELECT learner_id, objective_id, mean, variance,
                          stability_hours, exposures, last_seen_at,
                          next_review_at, evidence_mass, model_version
                   FROM objective_states WHERE learner_id = ?
                   ORDER BY objective_id""",
                (learner_id,),
            )
        ]
        misconception_beliefs = [
            dict(row)
            for row in connection.execute(
                """SELECT learner_id, misconception_id, log_odds,
                          evidence_count, last_seen_at, model_version
                   FROM misconception_beliefs WHERE learner_id = ?
                   ORDER BY misconception_id""",
                (learner_id,),
            )
        ]
        skill_families = [
            dict(row)
            for row in connection.execute(
                """SELECT learner_id, concept_id, family_id, kind,
                          first_unguided_correct_at,
                          last_unguided_correct_at,
                          delayed_unguided_correct_at
                   FROM learner_skill_families WHERE learner_id = ?
                   ORDER BY concept_id, family_id""",
                (learner_id,),
            )
        ]
        objective_families = [
            dict(row)
            for row in connection.execute(
                """SELECT learner_id, objective_id, family_id, kind,
                          first_unguided_correct_at,
                          last_unguided_correct_at,
                          delayed_unguided_correct_at
                   FROM learner_objective_families WHERE learner_id = ?
                   ORDER BY objective_id, family_id""",
                (learner_id,),
            )
        ]
        grid_rows = connection.execute(
            """SELECT learner_id, objective_id, posterior_schema_version,
                      algorithm, grid_id, codec, posterior_blob,
                      posterior_sha256, mean, variance, mastery_probability,
                      expected_competence, edge_mass,
                      mastery_probability_error_bound, evidence_mass,
                      acquisition_mass, model_version
               FROM objective_grid_states WHERE learner_id = ?
               ORDER BY objective_id""",
            (learner_id,),
        ).fetchall()

    objective_grid_states: list[dict[str, Any]] = []
    for row in grid_rows:
        posterior = decode_objective_posterior(
            bytes(row["posterior_blob"]),
            expected_digest=row["posterior_sha256"],
        )
        pending_observations = []
        for observation in posterior.pending_observations:
            semantic_observation = observation.as_payload()
            semantic_observation.pop("observation_id")
            pending_observations.append(semantic_observation)
        pending_observations.sort(
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        posterior_semantics = {
            "schema_version": row["posterior_schema_version"],
            "algorithm": row["algorithm"],
            "codec": row["codec"],
            "grid_id": row["grid_id"],
            "prior_mastery": posterior.prior_mastery,
            "prior_variance": posterior.prior_variance,
            "anchor_log_density": list(posterior.anchor_log_density),
            "current_log_density": list(posterior.log_density),
            "pending_observations": pending_observations,
            "committed_evidence_mass": posterior.committed_evidence_mass,
            "acquisition_mass": posterior.acquisition_mass,
        }
        objective_grid_states.append(
            {
                "learner_id": row["learner_id"],
                "objective_id": row["objective_id"],
                "posterior_schema_version": row[
                    "posterior_schema_version"
                ],
                "algorithm": row["algorithm"],
                "grid_id": row["grid_id"],
                "codec": row["codec"],
                "mean": row["mean"],
                "variance": row["variance"],
                "mastery_probability": row["mastery_probability"],
                "expected_competence": row["expected_competence"],
                "edge_mass": row["edge_mass"],
                "mastery_probability_error_bound": row[
                    "mastery_probability_error_bound"
                ],
                "evidence_mass": row["evidence_mass"],
                "acquisition_mass": row["acquisition_mass"],
                "model_version": row["model_version"],
                "posterior_semantics_sha256": canonical_hash(
                    posterior_semantics
                ),
            }
        )

    payload = {
        "signature_schema_version": SEMANTIC_PROJECTION_SIGNATURE_SCHEMA,
        "learner_id": learner["id"],
        "learner_revision": learner["revision"],
        "skill_states": skill_states,
        "objective_states": objective_states,
        "objective_grid_states": objective_grid_states,
        "misconception_beliefs": misconception_beliefs,
        "skill_families": skill_families,
        "objective_families": objective_families,
    }
    return {
        "schema_version": SEMANTIC_PROJECTION_SIGNATURE_SCHEMA,
        "sha256": canonical_hash(payload),
        "provenance_exclusions": [
            "skill_states.as_of_event_id",
            "objective_states.as_of_event_id",
            "objective_grid_states.as_of_event_id",
            "misconception_beliefs.as_of_event_id",
            "objective_grid_states.posterior_sha256",
            (
                "objective_grid_states.posterior_blob."
                "pending_observations[].observation_id"
            ),
        ],
        "retained_semantics": [
            "all scalar learner-state values and model versions",
            "complete exact posterior anchor and current densities",
            "pending observation values and multiplicity",
            "skill and objective family certification records",
            "misconception belief values",
            "learner revision and semantic timestamps",
        ],
    }


def projection_summary(
    profile: Mapping[str, Any],
    *,
    database: Database,
    learner_id: str,
) -> dict[str, Any]:
    """Summarize both legacy and objective projections without double counting.

    ``engine.profile`` deliberately overlays broad concepts with derived
    objective floors for routing.  Those derived rows are valuable to inspect,
    but they are not additional persisted evidence.  Aggregate evidence here is
    therefore taken from the two durable state/family tables.
    """

    skills = list(profile["skills"])
    objectives = list(profile.get("learning_objectives", ()))
    active_misconceptions = list(profile["active_misconceptions"])
    misconception_hypotheses = list(
        profile.get("misconception_hypotheses", active_misconceptions)
    )
    with database.read() as connection:
        persisted_skill_ids = {
            row["concept_id"]
            for row in connection.execute(
                "SELECT concept_id FROM skill_states WHERE learner_id = ?",
                (learner_id,),
            )
        }
        legacy_state = connection.execute(
            """SELECT COUNT(*) AS states,
                      COALESCE(SUM(evidence_mass), 0.0) AS evidence_mass
               FROM skill_states WHERE learner_id = ?""",
            (learner_id,),
        ).fetchone()
        objective_state = connection.execute(
            """SELECT COUNT(*) AS states,
                      COALESCE(SUM(evidence_mass), 0.0) AS evidence_mass
               FROM objective_states WHERE learner_id = ?""",
            (learner_id,),
        ).fetchone()
        legacy_family_count = connection.execute(
            "SELECT COUNT(*) AS n FROM learner_skill_families WHERE learner_id = ?",
            (learner_id,),
        ).fetchone()["n"]
        objective_family_count = connection.execute(
            """SELECT COUNT(*) AS n FROM learner_objective_families
               WHERE learner_id = ?""",
            (learner_id,),
        ).fetchone()["n"]
        objective_metadata = {
            row["objective_id"]: dict(row)
            for row in connection.execute(
                """SELECT state.objective_id, state.exposures,
                          state.last_seen_at, state.model_version,
                          grid.posterior_sha256, grid.edge_mass,
                          grid.mastery_probability_error_bound,
                          grid.acquisition_mass
                   FROM objective_states state
                   LEFT JOIN objective_grid_states grid
                     ON grid.learner_id = state.learner_id
                    AND grid.objective_id = state.objective_id
                   WHERE state.learner_id = ?
                   ORDER BY state.objective_id""",
                (learner_id,),
            )
        }

    skill_fields = (
        "name",
        "mastery",
        "expected_competence",
        "uncertainty",
        "stability_hours",
        "evidence_mass",
        "independent_families",
        "successful_retrieval_families",
        "observed_response_families",
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
    by_concept = {
        skill["concept_id"]: {
            **{key: skill.get(key) for key in skill_fields},
            "projection_kind": (
                "persisted_legacy"
                if skill["concept_id"] in persisted_skill_ids
                else "derived_objective_floor"
            ),
        }
        for skill in skills
    }
    objective_fields = (
        "name",
        "description",
        "operation",
        "evidence_type",
        "primary_concept_id",
        "supporting_concept_ids",
        "mastery",
        "expected_competence",
        "uncertainty",
        "stability_hours",
        "evidence_mass",
        "independent_families",
        "successful_retrieval_families",
        "observed_response_families",
        "delayed_retrievals",
        "operation_kinds",
        "active_misconception_probability",
        "prerequisite_mode",
        "prerequisite_objective_ids",
        "prerequisites_ready",
        "prerequisite_support",
        "state",
        "next_review_at",
    )
    by_objective: dict[str, dict[str, Any]] = {}
    for objective in objectives:
        objective_id = objective["objective_id"]
        metadata = objective_metadata.get(objective_id, {})
        exact_posterior = None
        if metadata.get("posterior_sha256") is not None:
            exact_posterior = {
                "provenance_bound_sha256": metadata["posterior_sha256"],
                "edge_mass": metadata["edge_mass"],
                "mastery_probability_error_bound": metadata[
                    "mastery_probability_error_bound"
                ],
                "acquisition_mass": metadata["acquisition_mass"],
            }
        by_objective[objective_id] = {
            **{key: objective.get(key) for key in objective_fields},
            "persisted": objective_id in objective_metadata,
            "exposures": metadata.get("exposures", 0),
            "last_seen_at": metadata.get("last_seen_at"),
            "model_version": metadata.get("model_version"),
            "exact_posterior": exact_posterior,
        }

    assessed_units = [
        {"id": concept_id, **by_concept[concept_id]}
        for concept_id in sorted(persisted_skill_ids & set(by_concept))
    ] + [
        {"id": objective_id, **by_objective[objective_id]}
        for objective_id in sorted(by_objective)
    ]
    observed_units = [
        unit for unit in assessed_units if float(unit["evidence_mass"] or 0.0) > 0
    ]
    commitment = _projection_commitment(database, learner_id)
    semantic_signature = semantic_projection_signature(database, learner_id)
    legacy_evidence_mass = float(legacy_state["evidence_mass"])
    objective_evidence_mass = float(objective_state["evidence_mass"])
    return {
        "skill_count": len(skills),
        "persisted_legacy_skill_count": int(legacy_state["states"]),
        "objective_count": len(objectives),
        "persisted_objective_count": int(objective_state["states"]),
        "observed_assessment_unit_count": len(observed_units),
        "legacy_evidence_mass": legacy_evidence_mass,
        "objective_evidence_mass": objective_evidence_mass,
        "total_evidence_mass": legacy_evidence_mass + objective_evidence_mass,
        "legacy_independent_families": int(legacy_family_count),
        "objective_independent_families": int(objective_family_count),
        "total_independent_families": int(
            legacy_family_count + objective_family_count
        ),
        "mean_mastery": (
            sum(float(unit["mastery"]) for unit in assessed_units)
            / len(assessed_units)
            if assessed_units
            else None
        ),
        "mean_observed_mastery": (
            sum(float(unit["mastery"]) for unit in observed_units)
            / len(observed_units)
            if observed_units
            else None
        ),
        "state_counts": dict(
            Counter(str(unit["state"]) for unit in assessed_units)
        ),
        "monitored_misconception_count": len(misconception_hypotheses),
        "active_misconception_count": len(active_misconceptions),
        "maximum_misconception_probability": max(
            (item["probability"] for item in misconception_hypotheses),
            default=None,
        ),
        "skills": by_concept,
        "learning_objectives": by_objective,
        "misconception_hypotheses": misconception_hypotheses,
        "active_misconceptions": active_misconceptions,
        "semantic_projection_signature": semantic_signature,
        "exact_event_projection_commitment": commitment,
        "aggregate_contract": (
            "Totals combine durable legacy skill rows with durable objective rows; "
            "derived broad objective floors are shown but never counted twice."
        ),
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
                    "surface_primary_concept_id": (
                        step.surface_primary_concept_id
                    ),
                    "surface_primary_concept_name": graph.concepts[
                        step.surface_primary_concept_id
                    ].name,
                    "evidence_anchor_concept_id": (
                        step.evidence_anchor_concept_id
                    ),
                    "evidence_anchor_concept_name": graph.concepts[
                        step.evidence_anchor_concept_id
                    ].name,
                    # Retained for old artifact readers; this is the surface
                    # primary and must not be treated as the evidence latent.
                    "primary_concept_id": step.primary_concept_id,
                    "learning_objective_id": step.learning_objective_id,
                    "primary_concept_name": graph.concepts[
                        step.primary_concept_id
                    ].name,
                    "learning_distance_to_root": distances.get(
                        step.evidence_anchor_concept_id
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
                    "objective_before": step.focus_objective_before,
                    "objective_after": step.focus_objective_after,
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
            "initial_focus_objective_id": episode.initial_focus_objective_id,
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
            "objective_focus_path": list(episode.objective_focus_path),
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
        "unique_objectives",
        "phase_counts",
        "response_time",
        "difficulty",
        "average_predicted_success",
        "continuity",
        "exploration",
        "remediation_questions",
        "cross_topic_questions",
        "cross_topic_definition",
        "outside_requested_topic_questions",
        "requested_topic_scope_definition",
        "topic_distribution",
        "evidence_delta",
        "concept_changes",
        "objective_changes",
        "adaptive_routing",
        "concept_performance",
        "objective_performance",
        "objective_evidence_scope",
        "family_evidence_definitions",
        "behavior_trace",
        "response_position_shadow",
        "diagnostic_findings",
        "diagnostic_contract",
    )
    return {key: report[key] for key in retained}


def capacity_and_demand_snapshot(
    database: Database, session: Mapping[str, Any]
) -> dict[str, Any]:
    """Expose unused evidence-family capacity and live authoring demand.

    Objective questions are anchored to their canonical objective owner rather
    than their surface primary concept.  Capacity keys also include the
    objective, matching the learner's actual independence bucket.
    """

    release_id = session["corpus_release_id"]
    if session.get("topic_id"):
        owned = database.topic_owned_concepts(
            session["topic_id"], release_id, include_descendants=True
        )
    else:
        owned = {session["root_concept_id"]}
    with database.read() as connection:
        rows = connection.execute(
            """SELECT surface.concept_id AS surface_concept_id,
                       question.id AS question_id, question.family_id,
                       question.kind, direct.objective_id,
                       objective.primary_concept_id AS objective_owner_concept_id
                FROM release_questions membership
                JOIN questions question ON question.id = membership.question_id
                JOIN question_concepts surface
                  ON surface.question_id = question.id
                 AND surface.role = 'primary'
                LEFT JOIN release_question_objectives direct
                  ON direct.release_id = membership.release_id
                 AND direct.question_id = question.id
                LEFT JOIN learning_objectives objective
                  ON objective.id = direct.objective_id
                WHERE membership.release_id = ?
                  AND membership.status IN ('approved', 'calibrated')
                ORDER BY question.id""",
            (release_id,),
        ).fetchall()
        used_rows = connection.execute(
            """SELECT question.family_id, decision.question_objective_id,
                      surface.concept_id AS surface_concept_id,
                      objective.primary_concept_id AS objective_owner_concept_id
               FROM decisions decision
               JOIN questions question ON question.id = decision.question_id
               JOIN question_concepts surface
                 ON surface.question_id = question.id AND surface.role = 'primary'
               LEFT JOIN learning_objectives objective
                 ON objective.id = decision.question_objective_id
               WHERE decision.session_id = ?
                 AND decision.invalidated_at IS NULL""",
            (session["id"],),
        ).fetchall()
        used_family_keys = {
            (
                f"objective:{row['question_objective_id']}|"
                f"family:{row['family_id']}"
                if row["question_objective_id"] is not None
                else (
                    f"concept:{row['surface_concept_id']}|"
                    f"family:{row['family_id']}"
                )
            )
            for row in used_rows
        }
        used_raw_families = {
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
    by_concept: dict[str, dict[str, set[str]]] = {
        concept_id: {
            "questions": set(),
            "families": set(),
            "verification_families": set(),
        }
        for concept_id in owned
    }
    by_objective: dict[str, dict[str, Any]] = {}
    for row in rows:
        objective_id = row["objective_id"]
        anchor = (
            row["objective_owner_concept_id"]
            if objective_id is not None
            else row["surface_concept_id"]
        )
        if anchor not in owned:
            continue
        family_key = (
            f"objective:{objective_id}|family:{row['family_id']}"
            if objective_id is not None
            else f"concept:{anchor}|family:{row['family_id']}"
        )
        values = by_concept[anchor]
        values["questions"].add(row["question_id"])
        values["families"].add(family_key)
        if row["kind"] in verification_kinds:
            values["verification_families"].add(family_key)
        if objective_id is not None:
            objective_values = by_objective.setdefault(
                objective_id,
                {
                    "primary_concept_id": anchor,
                    "questions": set(),
                    "families": set(),
                    "verification_families": set(),
                },
            )
            objective_values["questions"].add(row["question_id"])
            objective_values["families"].add(family_key)
            if row["kind"] in verification_kinds:
                objective_values["verification_families"].add(family_key)

    graph = database.get_graph(release_id)
    objective_names = {
        objective.id: objective.name
        for objective in database.get_learning_objectives(release_id)
    }
    concepts = []
    for concept_id in sorted(owned):
        values = by_concept[concept_id]
        families = values["families"]
        verification = values["verification_families"]
        used = families & used_family_keys
        concepts.append(
            {
                "concept_id": concept_id,
                "name": graph.concepts[concept_id].name,
                "approved_questions": len(values["questions"]),
                "independent_families": len(families),
                "verification_families": len(verification),
                "used_families": len(used),
                "remaining_families": len(families - used_family_keys),
                "remaining_verification_families": len(
                    verification - used_family_keys
                ),
            }
        )
    objectives = []
    for objective_id in sorted(by_objective):
        values = by_objective[objective_id]
        families = values["families"]
        verification = values["verification_families"]
        used = families & used_family_keys
        objectives.append(
            {
                "objective_id": objective_id,
                "name": objective_names.get(objective_id, objective_id),
                "primary_concept_id": values["primary_concept_id"],
                "approved_questions": len(values["questions"]),
                "independent_families": len(families),
                "verification_families": len(verification),
                "used_families": len(used),
                "remaining_families": len(families - used_family_keys),
                "remaining_verification_families": len(
                    verification - used_family_keys
                ),
            }
        )
    owned_family_keys = {
        family for values in by_concept.values() for family in values["families"]
    }
    return {
        "owned_concepts": concepts,
        "owned_objectives": objectives,
        "used_session_raw_families": len(used_raw_families),
        "used_session_evidence_families": len(used_family_keys),
        "remaining_owned_evidence_families": len(
            owned_family_keys - used_family_keys
        ),
        "denominator_contract": (
            "Objective/family pairs are distinct evidence buckets; legacy "
            "questions use canonical concept/family pairs."
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


def simulation_gap_records(
    report: SimulationReport,
    *,
    segment: str,
) -> list[dict[str, Any]]:
    """Return one uniform, inspectable record for every early exhaustion."""

    return [
        {
            "segment": segment,
            "step": gap.step_index,
            "phase": gap.phase.value,
            "focus_concept_id": gap.focus_concept_id,
            "focus_objective_id": gap.focus_objective_id,
            "focus_misconception_id": gap.focus_misconception_id,
            "category": gap.category,
            "message": gap.message,
        }
        for gap in report.gaps
    ]


def planned_check_status(
    failures: Iterable[str],
    gap_records: Iterable[Mapping[str, Any]],
) -> str:
    """Classify a planned probe without hiding capacity-limited completion."""

    if tuple(failures):
        return "failed"
    if tuple(gap_records):
        return "partial"
    return "passed"


def aggregate_audit_gaps(
    scenario_results: Iterable[Mapping[str, Any]],
    special_checks: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Combine ordinary scenario blockers and every planned-check gap."""

    result = [
        {"scenario": scenario["id"], **gap}
        for scenario in scenario_results
        for gap in scenario.get("gaps", ())
    ]
    result.extend(
        {"scenario": check["id"], **gap}
        for check in special_checks
        for gap in check.get("gap_records", ())
    )
    return result


def run_scenario(
    spec: ScenarioSpec,
    *,
    database_path: Path,
    corpus_path: Path,
    root_reference: str,
    steps: int,
    seed: int,
    replicate: bool,
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
        engine.profile(
            learner_id, root_concept_id=root_reference, now=report.ended_at
        ),
        database=database,
        learner_id=learner_id,
    )
    trace, trace_violations = serialize_trace(
        report, persisted_report, database, session
    )
    integrity = database.verify_integrity()
    invariant_failures = scenario_invariants(
        report, trace_violations, integrity
    )
    if not profile["exact_event_projection_commitment"][
        "matches_latest_event"
    ]:
        invariant_failures.append(
            "database projection differs from its latest event commitment"
        )

    exact_replay_report = ProjectionReplay(database).check(learner_id)
    exact_event_replay = {
        "checked": True,
        "ok": exact_replay_report["ok"],
        "response_count": exact_replay_report["response_count"],
        "stored_projection_matches_reconstruction": exact_replay_report[
            "source_projection_matches_replay"
        ],
        "event_commitment_matches_reconstruction": exact_replay_report[
            "commitment_matches_replay"
        ],
        "errors": exact_replay_report["errors"],
    }
    if not exact_event_replay["ok"]:
        invariant_failures.append(
            "exact same-event projection replay failed"
        )

    replication_result: dict[str, Any] | None = None
    if replicate:
        replica_path = database_path.with_name(
            database_path.stem + "-replica.db"
        )
        replica_database = Database(replica_path)
        replica_database.initialize()
        replica_database.import_corpus(
            *read_and_parse(corpus_path, include_catalog=True)
        )
        replica_engine = AdaptiveEngine(replica_database)
        replica_report = BehavioralSimulator(replica_engine).run(
            make_learner(spec.id, seed),
            learner_id=learner_id,
            root_concept_id=root_reference,
            policy_seed=seed,
            max_steps=steps,
            start_at=DEFAULT_START,
        )
        replica_profile = projection_summary(
            replica_engine.profile(
                learner_id,
                root_concept_id=root_reference,
                now=replica_report.ended_at,
            ),
            database=replica_database,
            learner_id=learner_id,
        )
        replica_integrity = replica_database.verify_integrity()
        behavior_matches = (
            report.behavior_signature() == replica_report.behavior_signature()
        )
        projection_matches = (
            profile["semantic_projection_signature"]["sha256"]
            == replica_profile["semantic_projection_signature"]["sha256"]
        )
        commitments_valid = bool(
            profile["exact_event_projection_commitment"][
                "matches_latest_event"
            ]
            and replica_profile["exact_event_projection_commitment"][
                "matches_latest_event"
            ]
        )
        if not behavior_matches:
            invariant_failures.append(
                "fresh-database semantic behavior replication diverged"
            )
        if not projection_matches:
            invariant_failures.append(
                "fresh-database semantic learner projection replication diverged"
            )
        if not commitments_valid:
            invariant_failures.append(
                "database projection differs from its latest event commitment"
            )
        if not replica_integrity["ok"]:
            invariant_failures.append("replica database integrity failed")
        replication_result = {
            "checked": True,
            "semantic_behavior_matches": behavior_matches,
            "semantic_projection_matches": projection_matches,
            "source_semantic_projection_sha256": profile[
                "semantic_projection_signature"
            ]["sha256"],
            "replica_semantic_projection_sha256": replica_profile[
                "semantic_projection_signature"
            ]["sha256"],
            "each_history_exact_commitment_valid": commitments_valid,
            "replica_database_integrity_ok": replica_integrity["ok"],
            "comparison_contract": (
                "Independent histories are compared only by behavior and "
                "provenance-free learner semantics. Exact event-bound "
                "commitments are validated within each history, never compared "
                "across histories."
            ),
        }

    summary = report.summary()
    result = {
        "id": spec.id,
        "behavior": spec.behavior,
        "hypothesis": spec.hypothesis,
        "synthetic_generator": generator_summary(learner),
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
        "exact_same_event_replay": exact_event_replay,
        "fresh_database_replication": (
            replication_result or {"checked": False}
        ),
        "invariant_failures": invariant_failures,
    }
    return result, graph_snapshot(
        database,
        root_reference=root_reference,
        release_id=release["release_id"],
    )


def run_position_habit_check(
    *,
    database_path: Path,
    corpus_path: Path,
    seed: int,
    steps: int = 16,
) -> dict[str, Any]:
    """Cross the conservative response-position threshold on the real engine.

    The ordinary Transformer scenario is intentionally allowed to expose a
    narrow-topic capacity gap, which can stop it one answer short of the
    twelve-observation statistical boundary.  This planned probe uses the
    broader Machine Learning curriculum, but otherwise runs the same selector,
    learner projection, immutable events, and report implementation.  The
    signal remains observational and cannot feed selection or mastery.
    """

    if steps < 12:
        raise ValueError("The position-habit probe requires at least 12 steps.")
    scenario, _topology = run_scenario(
        SCENARIO_BY_ID["fixed_option_bias"],
        database_path=database_path,
        corpus_path=corpus_path,
        root_reference="t_machine_learning",
        steps=steps,
        seed=seed,
        replicate=False,
    )
    shadow = scenario["session"]["response_position_shadow"]
    inference = shadow["inference"]
    dominant = inference.get("dominant_position")
    failures = list(scenario["invariant_failures"])
    if scenario["summary"]["attempted"] < 12:
        failures.append(
            "broad-curriculum probe did not reach the position-test boundary"
        )
    if not shadow["evidence"]["boundary_valid"]:
        failures.append("response-position evidence boundary was invalid")
    if inference["status"] != "position_concentration_signal":
        failures.append("fixed displayed-position habit did not produce a signal")
    if not isinstance(dominant, dict) or dominant.get("display_position") != 1:
        failures.append("the first displayed position was not the dominant signal")
    if not (
        shadow["observational_only"]
        and not shadow["affects_mastery"]
        and not shadow["affects_certification"]
        and not shadow["affects_selection"]
    ):
        failures.append("response-position diagnostic escaped its shadow boundary")
    gap_records = [
        {"segment": "broad_position_probe", **gap}
        for gap in scenario["gaps"]
    ]
    status = planned_check_status(failures, gap_records)
    return {
        "id": "display_position_shadow",
        "behavior": (
            "The learner selects displayed position one for every question in a "
            "broad curriculum until the exact family-wise test is estimable."
        ),
        "root_reference": "t_machine_learning",
        "planned_attempts": steps,
        "completed_attempts": scenario["summary"]["attempted"],
        "response_position_shadow": shadow,
        "projection": scenario["projection"],
        "trace": scenario["trace"],
        "gap_records": gap_records,
        "integrity": scenario["integrity"],
        "exact_same_event_replay": scenario["exact_same_event_replay"],
        "status": status,
        "ok": status == "passed",
        "failures": failures,
    }


def run_longitudinal_idempotency_check(
    *,
    database_path: Path,
    corpus_path: Path,
    root_reference: str,
    seed: int,
) -> dict[str, Any]:
    """Exercise two sessions and replay every answer command exactly once."""

    database = Database(database_path)
    database.initialize()
    release = database.import_corpus(
        *read_and_parse(corpus_path, include_catalog=True)
    )
    engine = AdaptiveEngine(database)
    simulator = BehavioralSimulator(engine)
    learner_id = "lab-longitudinal-idempotency"
    learner = SyntheticLearner(
        "longitudinal-idempotency",
        default_ability=0.86,
        default_objective_ability=0.86,
        slip_probability=0.0,
        guess_probability=0.0,
        forced_correctness=True,
        confidence_override=0.92,
        base_response_ms=4_000,
        response_model="ability_only",
        seed=seed + 911,
    )
    validate_profile_references(learner, database, release["release_id"])
    first = simulator.run(
        learner,
        learner_id=learner_id,
        root_concept_id=root_reference,
        policy_seed=seed + 101,
        max_steps=3,
        start_at=DEFAULT_START,
        trial_index=0,
        verify_idempotency=True,
    )
    second = simulator.run(
        learner,
        learner_id=learner_id,
        root_concept_id=root_reference,
        policy_seed=seed + 102,
        max_steps=3,
        start_at=first.ended_at + timedelta(days=7),
        trial_index=1,
        require_fresh_learner=False,
        verify_idempotency=True,
    )
    profile = projection_summary(
        engine.profile(
            learner_id,
            root_concept_id=root_reference,
            now=second.ended_at,
        ),
        database=database,
        learner_id=learner_id,
    )
    integrity = database.verify_integrity()
    with database.read() as connection:
        durable = dict(
            connection.execute(
                """SELECT learner.revision,
                          (SELECT COUNT(*) FROM sessions
                           WHERE learner_id = learner.id) AS sessions,
                          (SELECT COUNT(*) FROM attempts
                           WHERE learner_id = learner.id) AS attempts,
                          (SELECT COUNT(*) FROM events
                           WHERE learner_id = learner.id
                             AND event_type = 'ResponseSubmitted') AS responses
                   FROM learners learner WHERE learner.id = ?""",
                (learner_id,),
            ).fetchone()
        )
    expected_attempts = first.attempted + second.attempted
    failures = []
    if durable["sessions"] != 2:
        failures.append("expected exactly two durable sessions")
    if durable["attempts"] != expected_attempts:
        failures.append("idempotent retries changed the durable attempt count")
    if durable["responses"] != expected_attempts:
        failures.append("idempotent retries appended duplicate response events")
    if durable["revision"] != expected_attempts:
        failures.append("learner revision does not equal committed responses")
    if (
        first.idempotent_retries_verified != first.attempted
        or second.idempotent_retries_verified != second.attempted
    ):
        failures.append("not every answer completed an exact idempotent retry")
    if not profile["exact_event_projection_commitment"][
        "matches_latest_event"
    ]:
        failures.append("final projection commitment does not match the database")
    if not integrity["ok"]:
        failures.append("longitudinal database integrity failed")
    gap_records = [
        *simulation_gap_records(first, segment="first_session"),
        *simulation_gap_records(second, segment="second_session"),
    ]
    status = planned_check_status(failures, gap_records)
    return {
        "id": "multi_session_idempotency",
        "behavior": (
            "One learner runs two sessions seven days apart; every answer "
            "is immediately retried with the same command key."
        ),
        "first_session": first.summary(),
        "second_session": second.summary(),
        "durable_counts": durable,
        "expected_attempts": expected_attempts,
        "idempotent_retries_verified": (
            first.idempotent_retries_verified
            + second.idempotent_retries_verified
        ),
        "shared_family_count": len(
            set(first.coverage.observed_families)
            & set(second.coverage.observed_families)
        ),
        "planned_attempts": 6,
        "completed_attempts": expected_attempts,
        "gap_records": gap_records,
        "final_projection": profile,
        "integrity": {
            "ok": integrity["ok"],
            "event_count": integrity["event_count"],
            "stream_count": integrity["stream_count"],
            "errors": integrity["errors"],
        },
        "status": status,
        "ok": status == "passed",
        "failures": failures,
    }


def run_delayed_family_retrieval_check(
    *,
    database_path: Path,
    corpus_path: Path,
    seed: int,
    topic_reference: str = "t_generative_modeling",
) -> dict[str, Any]:
    """Force a due family revisit across sessions and inspect its certificate.

    A single session intentionally cannot reuse a family.  This probe creates
    enough one-answer sessions to exhaust the topic's distinct-family pool,
    spacing them beyond the model's maximum 30-day independence interval.  It
    therefore observes real policy behavior on a delayed family revisit instead
    of merely unit-testing the family ledger in isolation.
    """

    database = Database(database_path)
    database.initialize()
    release = database.import_corpus(
        *read_and_parse(corpus_path, include_catalog=True)
    )
    engine = AdaptiveEngine(database)
    simulator = BehavioralSimulator(engine)
    learner_id = "lab-delayed-family-retrieval"
    learner = SyntheticLearner(
        "delayed-family-retrieval",
        default_ability=0.90,
        default_objective_ability=0.90,
        slip_probability=0.0,
        guess_probability=0.0,
        forced_correctness=True,
        confidence_override=0.95,
        base_response_ms=4_000,
        response_model="ability_only",
        seed=seed + 1_701,
    )
    validate_profile_references(learner, database, release["release_id"])

    reports: list[SimulationReport] = []
    first = simulator.run(
        learner,
        learner_id=learner_id,
        root_concept_id=topic_reference,
        policy_seed=seed + 201,
        max_steps=1,
        start_at=DEFAULT_START,
        trial_index=0,
    )
    reports.append(first)
    # One more session than the declared family denominator guarantees a
    # revisit if every family is serviceable.  If some are not serviceable, the
    # revisit occurs earlier and exposes that difference in the artifact.
    session_count = first.coverage.eligible_families + 1
    for index in range(1, session_count):
        reports.append(
            simulator.run(
                learner,
                learner_id=learner_id,
                root_concept_id=topic_reference,
                policy_seed=seed + 201 + index,
                max_steps=1,
                start_at=DEFAULT_START + timedelta(days=45 * index),
                trial_index=index,
                require_fresh_learner=False,
            )
        )

    observed = [
        step.family_id
        for report in reports
        for step in report.steps
    ]
    family_counts = Counter(observed)
    repeated_families = sorted(
        family_id for family_id, count in family_counts.items() if count > 1
    )
    with database.read() as connection:
        certificate_rows = connection.execute(
            """SELECT family_id, delayed_unguided_correct_at,
                      'concept' AS dimension
                 FROM learner_skill_families WHERE learner_id = ?
               UNION ALL
               SELECT family_id, delayed_unguided_correct_at,
                      'objective' AS dimension
                 FROM learner_objective_families WHERE learner_id = ?
               ORDER BY family_id, dimension""",
            (learner_id, learner_id),
        ).fetchall()
    delayed_families = sorted(
        {
            row["family_id"]
            for row in certificate_rows
            if row["delayed_unguided_correct_at"] is not None
        }
    )
    integrity = database.verify_integrity()
    failures = []
    if any(report.attempted != 1 for report in reports):
        failures.append("a one-answer longitudinal session did not serve exactly once")
    if not repeated_families:
        failures.append("the due family pool was never revisited")
    missing_delayed = sorted(set(repeated_families) - set(delayed_families))
    if missing_delayed:
        failures.append(
            "revisited due families lacked delayed-retrieval certificates: "
            + ", ".join(missing_delayed)
        )
    if not integrity["ok"]:
        failures.append("delayed-family database integrity failed")
    gap_records = [
        gap
        for index, report in enumerate(reports, start=1)
        for gap in simulation_gap_records(
            report, segment=f"session_{index}"
        )
    ]
    status = planned_check_status(failures, gap_records)
    return {
        "id": "delayed_family_retrieval",
        "behavior": (
            "A consistently correct learner completes one answer per session; "
            "sessions are 45 days apart and continue past the topic's declared "
            "distinct-family denominator."
        ),
        "topic_reference": topic_reference,
        "session_count": session_count,
        "declared_eligible_families": first.coverage.eligible_families,
        "observed_sequence": observed,
        "family_attempt_counts": dict(sorted(family_counts.items())),
        "repeated_families": repeated_families,
        "delayed_certified_families": delayed_families,
        "certificate_rows": [dict(row) for row in certificate_rows],
        "planned_attempts": session_count,
        "completed_attempts": sum(report.attempted for report in reports),
        "gap_records": gap_records,
        "integrity": {
            "ok": integrity["ok"],
            "event_count": integrity["event_count"],
            "stream_count": integrity["stream_count"],
            "errors": integrity["errors"],
        },
        "status": status,
        "ok": status == "passed",
        "failures": failures,
    }


def run_misconception_recovery_check(
    *,
    database_path: Path,
    corpus_path: Path,
    seed: int,
    root_reference: str = "c_causal_masking",
    target_misconception_id: str = "m_mask_only_inference",
) -> dict[str, Any]:
    """Observe a named misconception emerge and then recede under transfer.

    The first session selects one specific authored distractor whenever it is
    offered.  Later, well-spaced sessions answer correctly until the hypothesis
    retires, with a four-session safety bound.  The engine—not this probe—chooses
    all questions, prerequisites, and phase transitions.
    """

    database = Database(database_path)
    database.initialize()
    release = database.import_corpus(
        *read_and_parse(corpus_path, include_catalog=True)
    )
    engine = AdaptiveEngine(database)
    simulator = BehavioralSimulator(engine)
    induction_steps = 8
    recovery_steps_per_session = 1
    learner_id = "lab-misconception-recovery"
    actor_name = "misconception-recovery-switch"
    inducing_actor = MisconceptionProbeLearner(
        actor_name,
        target_misconception_id,
        True,
    )
    recovery_actor = MisconceptionProbeLearner(
        actor_name,
        target_misconception_id,
        False,
    )
    misconception_ids = {
        item.id
        for item in database.get_misconceptions(
            release_id=release["release_id"]
        )
    }
    if target_misconception_id not in misconception_ids:
        raise RuntimeError(
            "Recovery probe target is absent from the imported release: "
            f"{target_misconception_id}"
        )

    induction = simulator.run(
        inducing_actor,
        learner_id=learner_id,
        root_concept_id=root_reference,
        policy_seed=seed + 301,
        max_steps=induction_steps,
        start_at=DEFAULT_START,
        trial_index=0,
    )
    induced_belief = database.get_misconception_beliefs(learner_id).get(
        target_misconception_id
    )
    induced_probability = induced_belief.probability if induced_belief else 0.10
    induced_profile = engine.profile(
        learner_id,
        root_concept_id=root_reference,
        now=induction.ended_at,
    )

    recovery_reports: list[SimulationReport] = []
    recovery_probabilities: list[float] = []
    prior_end = induction.ended_at
    for index in range(1, 5):
        report = simulator.run(
            recovery_actor,
            learner_id=learner_id,
            root_concept_id=root_reference,
            policy_seed=seed + 301 + index,
            max_steps=recovery_steps_per_session,
            start_at=prior_end + timedelta(days=45),
            trial_index=index,
            require_fresh_learner=False,
        )
        recovery_reports.append(report)
        prior_end = report.ended_at
        belief = database.get_misconception_beliefs(learner_id).get(
            target_misconception_id
        )
        recovery_probabilities.append(belief.probability if belief else 0.10)
        # Observe at least two independent follow-ups, then stop once the
        # named hypothesis falls below the routing threshold.
        if index >= 2 and recovery_probabilities[-1] < 0.35:
            break
    intermediate_probability = recovery_probabilities[0]
    final_belief = database.get_misconception_beliefs(learner_id).get(
        target_misconception_id
    )
    final_probability = final_belief.probability if final_belief else 0.10
    final_profile = engine.profile(
        learner_id,
        root_concept_id=root_reference,
        now=recovery_reports[-1].ended_at,
    )

    def target_steps(report: SimulationReport) -> list[dict[str, Any]]:
        result = []
        for step in report.steps:
            question = database.get_question(
                step.question_id,
                release_id=release["release_id"],
            )
            if target_misconception_id not in question.misconception_ids:
                continue
            result.append(
                {
                    "question_id": step.question_id,
                    "family_id": step.family_id,
                    "phase": step.phase_before.value,
                    "correct": step.actual_correct,
                }
            )
        return result

    induction_targets = target_steps(induction)
    recovery_targets = [target_steps(report) for report in recovery_reports]
    active_after_induction = {
        item["misconception_id"]
        for item in induced_profile["active_misconceptions"]
    }
    active_after_recovery = {
        item["misconception_id"]
        for item in final_profile["active_misconceptions"]
    }
    integrity = database.verify_integrity()
    failures = []
    induced_wrong_families = {
        step["family_id"]
        for step in induction_targets
        if not step["correct"]
    }
    recovered_correct_families = {
        step["family_id"]
        for target_group in recovery_targets
        for step in target_group
        if step["correct"]
    }
    if len(induced_wrong_families) < 2:
        failures.append("policy did not expose enough independent target distractors")
    if induced_probability < 0.35:
        failures.append("repeated named errors did not create an active hypothesis")
    if target_misconception_id not in active_after_induction:
        failures.append("induced hypothesis was absent from the learner profile")
    if len(recovered_correct_families) < 2:
        failures.append(
            "policy did not surface two independent corrective target families"
        )
    if intermediate_probability >= induced_probability:
        failures.append("first corrective session did not reduce the hypothesis")
    if final_probability >= intermediate_probability:
        failures.append("additional delayed correction did not further reduce it")
    if final_probability >= 0.35:
        failures.append("corrective transfer did not retire the active hypothesis")
    if target_misconception_id in active_after_recovery:
        failures.append("retired hypothesis remained active in the learner profile")
    if not integrity["ok"]:
        failures.append("misconception-recovery database integrity failed")
    gap_records = [
        *simulation_gap_records(induction, segment="induction"),
        *(
            gap
            for index, report in enumerate(recovery_reports, start=1)
            for gap in simulation_gap_records(
                report, segment=f"recovery_{index}"
            )
        ),
    ]
    status = planned_check_status(failures, gap_records)
    completed_attempts = induction.attempted + sum(
        report.attempted for report in recovery_reports
    )
    planned_attempts = induction_steps + (
        recovery_steps_per_session * len(recovery_reports)
    )
    return {
        "id": "misconception_recovery",
        "behavior": (
            "The learner repeatedly selects one named causal-mask distractor, "
            "then supplies one unassisted answer per bounded session 45 days apart."
        ),
        "root_reference": root_reference,
        "target_misconception_id": target_misconception_id,
        "probability_path": {
            "prior": 0.10,
            "after_induction": induced_probability,
            "after_first_recovery": intermediate_probability,
            "after_recovery_sessions": recovery_probabilities,
            "final": final_probability,
        },
        "target_evidence": {
            "induction": induction_targets,
            **{
                f"recovery_{index}": targets
                for index, targets in enumerate(recovery_targets, start=1)
            },
        },
        "independent_target_families": {
            "induction_wrong": sorted(induced_wrong_families),
            "recovery_correct": sorted(recovered_correct_families),
        },
        "active_after_induction": target_misconception_id
        in active_after_induction,
        "active_after_recovery": target_misconception_id
        in active_after_recovery,
        "capacity_gaps": {
            "induction": [gap.message for gap in induction.gaps],
            **{
                f"recovery_{index}": [gap.message for gap in report.gaps]
                for index, report in enumerate(recovery_reports, start=1)
            },
        },
        "planned_attempts": planned_attempts,
        "completed_attempts": completed_attempts,
        "attempted": completed_attempts,
        "gap_records": gap_records,
        "integrity": {
            "ok": integrity["ok"],
            "event_count": integrity["event_count"],
            "stream_count": integrity["stream_count"],
            "errors": integrity["errors"],
        },
        "status": status,
        "ok": status == "passed",
        "failures": failures,
    }


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
        "credible_over_missing_confidence_families",
        "deliberate_correct",
        "missing_confidence_correct",
        "projection.total_independent_families",
        ">",
        "Observed confidence is required before correct work certifies a family.",
    ),
    ComparisonSpec(
        "credible_over_missing_confidence_evidence",
        "deliberate_correct",
        "missing_confidence_correct",
        "projection.total_evidence_mass",
        ">",
        "Missing confidence remains discounted uncertain evidence.",
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
        "named_wrong_over_abstain_evidence",
        "confident_misconception",
        "uncertain_abstention",
        "projection.total_evidence_mass",
        ">",
        "A named wrong choice carries stronger negative evidence than explicit abstention.",
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
        "projection.learning_objectives.lo_causal_visibility.mastery",
        ">",
        "Targeted causal-masking errors lower that exact objective relative to credible work.",
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
        "--no-replication-check",
        action="store_true",
        help="skip the independent fresh-database semantic replication run",
    )
    result.add_argument(
        "--fail-on-hypothesis",
        action="store_true",
        help=(
            "exit 3 when a behavioral hypothesis is contradicted or a "
            "planned special check is incomplete"
        ),
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
            replicate=not arguments.no_replication_check,
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
    longitudinal = run_longitudinal_idempotency_check(
        database_path=database_directory / "multi_session_idempotency.db",
        corpus_path=arguments.corpus,
        root_reference=arguments.root,
        seed=arguments.seed,
    )
    invariant_failures.extend(
        {"scenario": longitudinal["id"], "failure": failure}
        for failure in longitudinal["failures"]
    )
    delayed_family = run_delayed_family_retrieval_check(
        database_path=database_directory / "delayed_family_retrieval.db",
        corpus_path=arguments.corpus,
        seed=arguments.seed,
    )
    invariant_failures.extend(
        {"scenario": delayed_family["id"], "failure": failure}
        for failure in delayed_family["failures"]
    )
    misconception_recovery = run_misconception_recovery_check(
        database_path=database_directory / "misconception_recovery.db",
        corpus_path=arguments.corpus,
        seed=arguments.seed,
    )
    invariant_failures.extend(
        {"scenario": misconception_recovery["id"], "failure": failure}
        for failure in misconception_recovery["failures"]
    )
    position_habit = run_position_habit_check(
        database_path=database_directory / "display_position_shadow.db",
        corpus_path=arguments.corpus,
        seed=arguments.seed + 401,
    )
    invariant_failures.extend(
        {"scenario": position_habit["id"], "failure": failure}
        for failure in position_habit["failures"]
    )
    special_checks = (
        longitudinal,
        delayed_family,
        misconception_recovery,
        position_habit,
    )
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
    blockers = aggregate_audit_gaps(results, special_checks)
    planned_check_statuses = {
        check["id"]: check["status"] for check in special_checks
    }
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
            "fresh_database_semantic_replication": (
                not arguments.no_replication_check
            ),
            "feedback_protocol": SIMULATION_FEEDBACK_PROTOCOL_VERSION,
            "scenario_ids": selected_ids,
            "position_habit_probe_root": "t_machine_learning",
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
        "longitudinal_check": longitudinal,
        "delayed_family_check": delayed_family,
        "misconception_recovery_check": misconception_recovery,
        "display_position_shadow_check": position_habit,
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
            "longitudinal_idempotency_ok": longitudinal["ok"],
            "delayed_family_retrieval_ok": delayed_family["ok"],
            "misconception_recovery_ok": misconception_recovery["ok"],
            "display_position_shadow_ok": position_habit["ok"],
            "planned_check_statuses": planned_check_statuses,
            "all_planned_checks_complete": all(
                status == "passed"
                for status in planned_check_statuses.values()
            ),
            "partial_planned_checks": sorted(
                check_id
                for check_id, status in planned_check_statuses.items()
                if status == "partial"
            ),
            "failed_planned_checks": sorted(
                check_id
                for check_id, status in planned_check_statuses.items()
                if status == "failed"
            ),
            "total_answers": sum(
                result["summary"]["attempted"] for result in results
            )
            + longitudinal["completed_attempts"]
            + delayed_family["completed_attempts"]
            + misconception_recovery["completed_attempts"]
            + position_habit["completed_attempts"],
            "all_integrity_checks_ok": all(
                result["integrity"]["ok"] for result in results
            )
            and longitudinal["integrity"]["ok"]
            and delayed_family["integrity"]["ok"]
            and misconception_recovery["integrity"]["ok"]
            and position_habit["integrity"]["ok"],
            "all_exact_same_event_replays_ok": all(
                result["exact_same_event_replay"]["ok"]
                for result in results
            )
            and position_habit["exact_same_event_replay"]["ok"],
            "all_fresh_database_semantic_replications_match": all(
                not result["fresh_database_replication"].get("checked")
                or (
                    result["fresh_database_replication"][
                        "semantic_behavior_matches"
                    ]
                    and result["fresh_database_replication"][
                        "semantic_projection_matches"
                    ]
                    and result["fresh_database_replication"][
                        "each_history_exact_commitment_valid"
                    ]
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
        and (
            artifact["audit"]["contradicted_hypotheses"]
            or not artifact["audit"]["all_planned_checks_complete"]
        )
    ):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
