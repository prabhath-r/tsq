#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0

"""Falsify or support TSQ's ability to localize objective-level weaknesses.

The laboratory drives the production engine through multiple spaced sessions.
Its deterministic answer policy is deliberately simple: every distinction is
answered correctly except one declared objective.  This is not human
calibration.  It is an identifiability probe: if TSQ cannot discover a stark,
stable weakness under this controlled pattern, a richer learner model will not
rescue the routing policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tsq.corpus import corpus_source_digest, read_and_parse  # noqa: E402
from tsq.engine import AdaptiveEngine  # noqa: E402
from tsq.learner import (  # noqa: E402
    FamilyResponseRecord,
    LearnerModel,
)
from tsq.models import Presentation  # noqa: E402
from tsq.objective_posterior import (  # noqa: E402
    LikelihoodObservation,
    ObjectivePosterior,
)
from tsq.policy import AdaptivePolicy  # noqa: E402
from tsq.replay import ProjectionReplay  # noqa: E402
from tsq.simulation import (  # noqa: E402
    BehavioralSimulator,
    SIMULATION_FEEDBACK_PROTOCOL_VERSION,
    SyntheticAnswer,
)
from tsq.store import Database  # noqa: E402


LAB_VERSION = "objective-discovery-lab-v5"
DEFAULT_CORPUS = PROJECT_ROOT / "corpus"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "experiments" / "results" / "objective_discovery_lab.json"
)
DEFAULT_TOPIC = "t_transformers"
DEFAULT_START = datetime(2105, 1, 5, 9, 0, tzinfo=timezone.utc)
DEFAULT_TARGETS = (
    "lo_attention_logit_scaling",
    "lo_causal_visibility",
    "lo_incremental_kv_cache",
    "lo_transformer_sublayer_composition",
)


class DiscoveryInvariantError(RuntimeError):
    """A durable/replay invariant failed during the laboratory run."""


def schedule_regime(
    weak_schedule: Sequence[bool],
) -> tuple[str, int | None]:
    """Classify a deterministic intervention without inventing a recovery.

    Uniform weak and uniform strong schedules are controls.  A recovery run
    must be one weak prefix followed by one strong suffix; switching back to
    weak behavior would make a single trough-to-final statistic meaningless.
    """
    if not weak_schedule:
        raise ValueError("The objective behavior schedule cannot be empty.")
    if any(type(value) is not bool for value in weak_schedule):
        raise ValueError("Every objective behavior schedule value must be boolean.")
    first_strong = next(
        (
            index
            for index, target_is_weak in enumerate(weak_schedule)
            if not target_is_weak
        ),
        None,
    )
    if first_strong is None:
        return "weak_control", None
    if first_strong == 0:
        if any(weak_schedule):
            raise ValueError(
                "A recovery schedule cannot switch from strong back to weak."
            )
        return "strong_control", None
    if any(weak_schedule[first_strong:]):
        raise ValueError(
            "A recovery schedule must be one weak prefix followed by one "
            "strong suffix."
        )
    return "weak_to_strong_recovery", first_strong


@dataclass(frozen=True, slots=True)
class ObjectivePatternLearner:
    """Answer against one objective while remaining strong elsewhere."""

    name: str
    target_objective_id: str
    target_is_weak: bool
    confidence: float = 0.95
    response_ms: int = 4_000
    rule: str = "objective-localized"

    def answer(
        self,
        presentation: Presentation,
        *,
        simulation_seed: int,
        trial_index: int,
        encounter: int,
    ) -> SyntheticAnswer:
        del simulation_seed, trial_index, encounter
        question = presentation.question
        diagnostic = [
            option
            for option in question.options
            if not option.correct
            and option.diagnostic_objective_id == self.target_objective_id
        ]
        # The localization experiment changes one *primary* latent objective.
        # A distractor may diagnose that objective from an item calibrated on
        # another objective, but deliberately missing that whole item would
        # also be negative evidence about its primary objective.  Mixing those
        # two interventions made an earlier laboratory version damage adjacent
        # objectives while claiming isolation.  Diagnostic transfer is a
        # separate hypothesis and must be tested separately.
        target_bearing = question.objective_id == self.target_objective_id
        correct = not (self.target_is_weak and target_bearing)
        if correct:
            selected = question.correct_option
        elif diagnostic:
            selected = sorted(diagnostic, key=lambda option: option.id)[0]
        else:
            named = sorted(
                (
                    option
                    for option in question.options
                    if not option.correct and option.misconception_id
                ),
                key=lambda option: option.id,
            )
            distractors = named or sorted(
                (option for option in question.options if not option.correct),
                key=lambda option: option.id,
            )
            if not distractors:
                raise DiscoveryInvariantError(
                    f"Question {question.id} has no incorrect option."
                )
            selected = distractors[0]
        return SyntheticAnswer(
            selected_option_id=selected.id,
            correct=correct,
            ground_truth_probability=0.02 if not correct else 0.98,
            confidence=self.confidence,
            response_ms=self.response_ms,
            hint_count=0,
        )


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def corpus_hash(path: Path) -> str:
    return corpus_source_digest(path)


def bounded_family_recovery_probe(
    *,
    corpus: Path = DEFAULT_CORPUS,
    objective_id: str = DEFAULT_TARGETS[0],
    max_correct_retests: int = 3,
    spacing_days: int = 45,
    required_recovery_fraction: float = 0.75,
) -> dict[str, Any]:
    """Try to falsify bounded recovery for every released family in an objective.

    Each case applies one credible wrong answer, followed by credible correct
    retests of that *same* family at the production spacing interval.  It uses
    the production exact posterior, status evidence weight, bounded family
    power, retention transition, and correct-feedback transition.  The probe
    therefore catches the historical failure mode where a single wrong family
    could never be reversed even by its entire admissible positive-evidence
    tail.

    This is intentionally a fast mechanism probe, not a routing claim.  The
    longitudinal engine case below separately checks whether policy actually
    returns to enough target families after the behavior changes.
    """

    if max_correct_retests <= 0:
        raise ValueError("max_correct_retests must be positive.")
    if spacing_days <= 0:
        raise ValueError("spacing_days must be positive.")
    if not 0.0 < required_recovery_fraction <= 1.0:
        raise ValueError("required_recovery_fraction must be in (0, 1].")

    questions = [
        question
        for question in read_and_parse(corpus)[4]
        if question.objective_id == objective_id
        and question.status.eligible_for_adaptation
    ]
    if not questions:
        raise DiscoveryInvariantError(
            f"No adaptation-eligible questions own objective {objective_id}."
        )

    model = LearnerModel()
    started_at = DEFAULT_START
    cases: list[dict[str, Any]] = []
    for question in sorted(questions, key=lambda item: item.id):
        if question.objective is None:
            raise DiscoveryInvariantError(
                f"Objective-aware question {question.id} lacks objective metadata."
            )
        base_weight = question.status.evidence_weight
        initial_state = model.initial_objective_state(
            "bounded-family-recovery-probe", question.objective
        )
        prior = ObjectivePosterior.from_prior(
            question.objective.prior_mastery
        )
        prior_mastery = prior.mastery_probability
        family_records: list[FamilyResponseRecord] = []

        def observation(
            sequence: int,
            *,
            correct: bool,
            evidence_power: float,
        ) -> LikelihoodObservation:
            return LikelihoodObservation(
                observation_id=f"{question.id}:probe:{sequence}",
                family_id=question.family_id,
                difficulty=question.difficulty,
                discrimination=question.discrimination,
                guess_rate=question.guess_rate,
                slip_rate=question.slip_rate,
                option_count=len(question.options),
                correct=correct,
                evidence_power=evidence_power,
            )

        wrong_power = model.spacing_aware_family_evidence_power(
            prior_records=family_records,
            occurred_at=started_at,
            credible=True,
        )
        posterior = prior.with_observation(
            observation(
                0,
                correct=False,
                evidence_power=base_weight * wrong_power.power,
            )
        )
        family_records.append(
            FamilyResponseRecord(occurred_at=started_at, credible=True)
        )
        trough_mastery = posterior.mastery_probability
        path = [trough_mastery]
        power_path = [
            {
                "response": "wrong",
                "family_power": wrong_power.power,
                "base_family_power": wrong_power.base_power,
                "renewal_power": wrong_power.renewal_power,
                "effective_evidence_power": base_weight * wrong_power.power,
            }
        ]
        recovered_after: int | None = None
        recovery_fraction = 0.0

        for retest in range(1, max_correct_retests + 1):
            occurred_at = started_at + timedelta(
                days=spacing_days * retest
            )
            posterior = posterior.apply_retention(
                elapsed_hours=float(spacing_days * 24),
                stability_hours=initial_state.stability_hours,
            )
            family_power = model.spacing_aware_family_evidence_power(
                prior_records=family_records,
                occurred_at=occurred_at,
                credible=True,
            )
            effective_power = base_weight * family_power.power
            posterior = posterior.with_observation(
                observation(
                    retest,
                    correct=True,
                    evidence_power=effective_power,
                )
            ).apply_correct_feedback(effective_power)
            family_records.append(
                FamilyResponseRecord(occurred_at=occurred_at, credible=True)
            )
            path.append(posterior.mastery_probability)
            power_path.append(
                {
                    "response": "correct",
                    "family_power": family_power.power,
                    "base_family_power": family_power.base_power,
                    "renewal_power": family_power.renewal_power,
                    "effective_evidence_power": effective_power,
                }
            )
            recoverable_gap = prior_mastery - trough_mastery
            recovery_fraction = (
                (posterior.mastery_probability - trough_mastery)
                / recoverable_gap
                if recoverable_gap > 0.0
                else 0.0
            )
            if recovery_fraction >= required_recovery_fraction:
                recovered_after = retest
                break

        cases.append(
            {
                "question_id": question.id,
                "family_id": question.family_id,
                "status": question.status.value,
                "prior_mastery": prior_mastery,
                "after_wrong_mastery": trough_mastery,
                "mastery_path": path,
                "power_path": power_path,
                "recovery_fraction": recovery_fraction,
                "recovered_after_correct_retests": recovered_after,
                "recovery_observed": recovered_after is not None,
            }
        )

    failures = [
        {
            "question_id": case["question_id"],
            "family_id": case["family_id"],
            "recovery_fraction": case["recovery_fraction"],
        }
        for case in cases
        if not case["recovery_observed"]
    ]
    return {
        "objective_id": objective_id,
        "learner_model_version": model.model_version,
        "max_correct_retests": max_correct_retests,
        "spacing_days": spacing_days,
        "required_recovery_fraction": required_recovery_fraction,
        "case_count": len(cases),
        "cases": cases,
        "failed_cases": failures,
        "all_families_recovered": not failures,
        "criterion": (
            "After one credible wrong response, every adaptation-eligible item "
            "family must recover at least the declared fraction of its "
            "cold-start mastery gap within the bounded number of credible, "
            "spaced, same-family correct retests."
        ),
        "interpretation_boundary": (
            "This directly falsifies posterior/family-power recoverability. "
            "It does not prove that policy will route the retests or that a "
            "human learned from feedback."
        ),
    }


def latest_session_id(database: Database, learner_id: str) -> str:
    with database.read() as connection:
        row = connection.execute(
            """SELECT id FROM sessions WHERE learner_id=?
               ORDER BY created_at DESC, id DESC LIMIT 1""",
            (learner_id,),
        ).fetchone()
    if row is None:
        raise DiscoveryInvariantError(
            f"No session was persisted for {learner_id}."
        )
    return str(row["id"])


def persistent_gap_episode_spends(
    database: Database,
    *,
    session_id: str,
) -> list[dict[str, Any]]:
    """Expose durable v13 episode markers in immutable response order."""

    with database.read() as connection:
        rows = connection.execute(
            """SELECT decision.question_id,
                      decision.question_objective_id,
                      decision.rationale,
                      decision.policy_version,
                      attempt.family_id,
                      event.stream_version
               FROM attempts attempt
               JOIN decisions decision
                 ON decision.id = attempt.decision_id
               JOIN events event ON event.event_id = attempt.event_id
               WHERE attempt.session_id = ?
                 AND event.event_type = 'ResponseSubmitted'
               ORDER BY event.stream_version""",
            (session_id,),
        ).fetchall()
    spends: list[dict[str, Any]] = []
    for response_index, row in enumerate(rows, start=1):
        marker = AdaptivePolicy._persistent_gap_marker(
            rationale=row["rationale"],
            policy_version=row["policy_version"],
            decision_objective_id=row["question_objective_id"],
        )
        if marker is None:
            continue
        spends.append(
            {
                "response_index": response_index,
                "stream_version": row["stream_version"],
                "question_id": row["question_id"],
                "objective_id": marker["objective_id"],
                "family_id": row["family_id"],
                "spend": marker["spend"],
                "budget": marker["budget"],
                "opened_due_at": marker["opened_due_at"].isoformat(),
                "mastery_probability": marker["mastery_probability"],
                "cold_start_mastery_probability": marker[
                    "cold_start_mastery_probability"
                ],
            }
        )
    return spends


def persistent_gap_episode_audit(
    sessions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Check the observable two-family/interleaving contract."""

    violations: list[dict[str, Any]] = []
    episode_count = 0
    spend_count = 0
    maximum_spends = 0
    for session in sessions:
        by_objective: dict[str, list[dict[str, Any]]] = {}
        for spend in session["persistent_gap_episode_spends"]:
            by_objective.setdefault(spend["objective_id"], []).append(spend)
        for objective_id, spends in sorted(by_objective.items()):
            episode_count += 1
            spend_count += len(spends)
            maximum_spends = max(maximum_spends, len(spends))
            expected_spends = list(range(1, len(spends) + 1))
            actual_spends = [spend["spend"] for spend in spends]
            families = [spend["family_id"] for spend in spends]
            interleaved = True
            if len(spends) == 2:
                first_index = spends[0]["response_index"]
                second_index = spends[1]["response_index"]
                interleaved = any(
                    step["objective_id"] != objective_id
                    for step in session["trace"][
                        first_index:second_index - 1
                    ]
                )
            if (
                len(spends) > 2
                or actual_spends != expected_spends
                or len(families) != len(set(families))
                or not interleaved
                or any(spend["budget"] != 2 for spend in spends)
            ):
                violations.append(
                    {
                        "session_index": session["index"],
                        "objective_id": objective_id,
                        "spends": spends,
                        "interleaved": interleaved,
                    }
                )
    return {
        "episode_count": episode_count,
        "spend_count": spend_count,
        "maximum_spends_per_session_objective": maximum_spends,
        "violations": violations,
        "all_contracts_passed": not violations,
        "criterion": (
            "Each session/objective episode spends at most two distinct "
            "serviceable families in sequence, with at least one response to "
            "another objective between spend one and spend two."
        ),
    }


def objective_snapshot(
    engine: AdaptiveEngine,
    learner_id: str,
    *,
    topic: str,
    now: datetime,
) -> dict[str, dict[str, Any]]:
    profile = engine.profile(learner_id, root_concept_id=topic, now=now)
    return {
        row["objective_id"]: {
            "mastery_probability": row["mastery_probability"],
            "expected_competence": row["expected_competence"],
            "uncertainty": row["uncertainty"],
            "evidence_mass": row["evidence_mass"],
            "observed_response_families": row[
                "observed_response_families"
            ],
            "successful_retrieval_families": row[
                "successful_retrieval_families"
            ],
            "delayed_retrievals": row["delayed_retrievals"],
            "active_misconception_probability": row[
                "active_misconception_probability"
            ],
            "state": row["state"],
        }
        for row in profile.get("learning_objectives", [])
    }


def discovery_metrics(
    snapshots: Sequence[dict[str, dict[str, Any]]],
    *,
    target_objective_id: str,
    baseline_snapshot: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if not snapshots or target_objective_id not in snapshots[-1]:
        raise DiscoveryInvariantError(
            f"Target objective {target_objective_id} is missing from profile."
        )
    final = snapshots[-1]
    target = final[target_objective_id]
    ordered = sorted(
        final,
        key=lambda objective_id: (
            final[objective_id]["mastery_probability"],
            objective_id,
        ),
    )
    target_rank = ordered.index(target_objective_id) + 1
    observed_strong = [
        row["mastery_probability"]
        for objective_id, row in final.items()
        if objective_id != target_objective_id
        and row["observed_response_families"] > 0
    ]
    median_observed_strong = (
        statistics.median(observed_strong) if observed_strong else None
    )
    target_path = [
        snapshot[target_objective_id]["mastery_probability"]
        for snapshot in snapshots
    ]
    first_detected_session = next(
        (
            index + 1
            for index, snapshot in enumerate(snapshots)
            if snapshot[target_objective_id]["observed_response_families"] >= 2
            and (
                1
                + sum(
                    row["mastery_probability"]
                    < snapshot[target_objective_id]["mastery_probability"]
                    for objective_id, row in snapshot.items()
                    if objective_id != target_objective_id
                )
            )
            <= 2
        ),
        None,
    )
    separated = bool(
        median_observed_strong is not None
        and target["mastery_probability"] + 0.05
        < median_observed_strong
    )
    non_target_regressions = [
        {
            "objective_id": objective_id,
            "baseline_mastery": baseline_snapshot[objective_id][
                "mastery_probability"
            ],
            "final_mastery": row["mastery_probability"],
            "change": (
                row["mastery_probability"]
                - baseline_snapshot[objective_id]["mastery_probability"]
            ),
            "observed_response_families": row[
                "observed_response_families"
            ],
        }
        for objective_id, row in sorted(final.items())
        if objective_id != target_objective_id
        and row["observed_response_families"] > 0
        and objective_id in baseline_snapshot
        and row["mastery_probability"] + 1e-8
        < baseline_snapshot[objective_id]["mastery_probability"]
    ]
    specificity_passed = not non_target_regressions
    hypothesis_passed = bool(
        target["observed_response_families"] >= 2
        and target_rank <= 2
        and separated
        and specificity_passed
    )
    return {
        "target_objective_id": target_objective_id,
        "target_rank_lowest_first": target_rank,
        "objective_count": len(final),
        "target_mastery_path": target_path,
        "target_final": target,
        "median_observed_strong_mastery": median_observed_strong,
        "separation": (
            None
            if median_observed_strong is None
            else median_observed_strong
            - target["mastery_probability"]
        ),
        "first_detected_session": first_detected_session,
        "specificity_passed": specificity_passed,
        "non_target_regressions": non_target_regressions,
        "hypothesis_passed": hypothesis_passed,
        "criterion": (
            "At least two target families, target among the two lowest "
            "objective mastery probabilities, and at least 0.05 below the "
            "median observed non-target objective, with no observed non-target "
            "objective falling below its cold-start mastery prior."
        ),
    }


def run_case(
    *,
    database_path: Path,
    corpus: Path,
    topic: str,
    target_objective_id: str,
    weak_schedule: Sequence[bool],
    steps_per_session: int,
    seed: int,
) -> dict[str, Any]:
    regime, recovery_boundary = schedule_regime(weak_schedule)
    database = Database(database_path)
    database.initialize()
    release = database.import_corpus(
        *read_and_parse(corpus, include_catalog=True)
    )
    known_objectives = {
        objective.id
        for objective in database.get_learning_objectives(
            release["release_id"]
        )
    }
    if target_objective_id not in known_objectives:
        raise DiscoveryInvariantError(
            f"Unknown target objective: {target_objective_id}"
        )
    engine = AdaptiveEngine(database)
    simulator = BehavioralSimulator(engine)
    learner_id = "discovery-" + target_objective_id.replace("lo_", "")
    stable_profile_name = f"{target_objective_id}-longitudinal"
    engine.create_learner(
        learner_id, f"simulation:{stable_profile_name}"
    )
    baseline_snapshot = objective_snapshot(
        engine,
        learner_id,
        topic=topic,
        now=DEFAULT_START,
    )
    sessions: list[dict[str, Any]] = []
    snapshots: list[dict[str, dict[str, Any]]] = []

    for session_index, target_is_weak in enumerate(weak_schedule):
        started_at = DEFAULT_START + timedelta(days=45 * session_index)
        learner = ObjectivePatternLearner(
            # The durable learner identity must keep one display name across
            # the weak-to-recovered schedule.  The changing latent pattern is
            # recorded separately in each session artifact.
            name=stable_profile_name,
            target_objective_id=target_objective_id,
            target_is_weak=target_is_weak,
        )
        report = simulator.run(
            learner,
            learner_id=learner_id,
            root_concept_id=topic,
            policy_seed=seed + 101 * session_index,
            max_steps=steps_per_session,
            start_at=started_at,
            inter_item_delay=timedelta(minutes=4),
            trial_index=session_index,
            require_fresh_learner=session_index == 0,
            verify_idempotency=session_index == 0,
        )
        session_id = latest_session_id(database, learner_id)
        engine.end_session(
            session_id,
            completed=True,
            reason="objective discovery laboratory boundary",
            idempotency_key=(
                f"objective-discovery-end-{target_objective_id}-{session_index}"
            ),
            now=report.ended_at,
        )
        snapshot_at = report.ended_at
        snapshot = objective_snapshot(
            engine, learner_id, topic=topic, now=snapshot_at
        )
        snapshots.append(snapshot)
        sessions.append(
            {
                "index": session_index + 1,
                "target_is_weak": target_is_weak,
                "started_at": started_at.isoformat(),
                "ended_at": report.ended_at.isoformat(),
                "attempted": report.attempted,
                "correct": report.correct,
                "gaps": [
                    {
                        "step": gap.step_index,
                        "phase": gap.phase.value,
                        "category": gap.category,
                        "message": gap.message,
                    }
                    for gap in report.gaps
                ],
                "remediation_episodes": [
                    {
                        "focus_objective_id": (
                            episode.initial_focus_objective_id
                        ),
                        "families": list(episode.family_ids),
                        "outcome": episode.outcome,
                    }
                    for episode in report.focus_episodes
                ],
                "trace": [
                    {
                        "step": step.index,
                        "question_id": step.question_id,
                        "family_id": step.family_id,
                        "objective_id": step.learning_objective_id,
                        "correct": step.actual_correct,
                        "phase_before": step.phase_before.value,
                        "phase_after": step.phase_after.value,
                        "focus_objective_after": (
                            step.focus_objective_after
                        ),
                    }
                    for step in report.steps
                ],
                "persistent_gap_episode_spends": (
                    persistent_gap_episode_spends(
                        database, session_id=session_id
                    )
                ),
                "objective_snapshot": snapshot,
            }
        )

    integrity = database.verify_integrity()
    replay = ProjectionReplay(database).check(learner_id)
    if not integrity["ok"]:
        raise DiscoveryInvariantError(
            "Integrity failure: " + "; ".join(integrity["errors"][:5])
        )
    if not replay["ok"]:
        raise DiscoveryInvariantError(
            "Projection replay failure: "
            + "; ".join(replay["errors"][:5])
        )
    metrics = discovery_metrics(
        snapshots,
        target_objective_id=target_objective_id,
        baseline_snapshot=baseline_snapshot,
    )
    complete_trace = [
        step for session in sessions for step in session["trace"]
    ]
    contract_violations = [
        {
            "question_id": step["question_id"],
            "objective_id": step["objective_id"],
            "family_id": step["family_id"],
        }
        for step in complete_trace
        if not step["correct"]
        and step["objective_id"] != target_objective_id
    ]
    if contract_violations:
        raise DiscoveryInvariantError(
            "The objective-localized learner answered outside its declared "
            f"target: {contract_violations[:3]}"
        )
    direct_target_attempts = [
        step
        for step in complete_trace
        if step["objective_id"] == target_objective_id
    ]
    metrics["direct_target_attempt_count"] = len(direct_target_attempts)
    metrics["direct_target_attempt_fraction"] = (
        len(direct_target_attempts) / len(complete_trace)
        if complete_trace
        else 0.0
    )
    metrics["direct_target_family_count"] = len(
        {step["family_id"] for step in direct_target_attempts}
    )
    metrics["behavior_contract_violations"] = contract_violations
    episode_audit = persistent_gap_episode_audit(sessions)
    if not episode_audit["all_contracts_passed"]:
        raise DiscoveryInvariantError(
            "Persistent-gap episode contract failure: "
            + repr(episode_audit["violations"][:3])
        )
    certificate_violations = [
        {
            "objective_id": objective_id,
            "observed_response_families": row[
                "observed_response_families"
            ],
            "successful_retrieval_families": row[
                "successful_retrieval_families"
            ],
        }
        for objective_id, row in sorted(snapshots[-1].items())
        if row["successful_retrieval_families"]
        > row["observed_response_families"]
    ]
    if certificate_violations:
        raise DiscoveryInvariantError(
            "Retrieval-family certificates exceed distinct observed families: "
            + repr(certificate_violations[:3])
        )
    target_path = metrics["target_mastery_path"]
    recovery: dict[str, Any] | None = None
    if recovery_boundary is not None:
        first_recovered = recovery_boundary
        induction_steps = [
            step
            for session in sessions[:first_recovered]
            for step in session["trace"]
            if step["objective_id"] == target_objective_id
            and not step["correct"]
        ]
        recovery_steps = [
            step
            for session in sessions[first_recovered:]
            for step in session["trace"]
            if step["objective_id"] == target_objective_id
            and step["correct"]
        ]
        induction_families = sorted(
            {step["family_id"] for step in induction_steps}
        )
        recovery_families = sorted(
            {step["family_id"] for step in recovery_steps}
        )
        baseline_mastery = baseline_snapshot[target_objective_id][
            "mastery_probability"
        ]
        induction_trough = min(target_path[:first_recovered])
        final_mastery = target_path[-1]
        recoverable_gap = max(0.0, baseline_mastery - induction_trough)
        recovery_progress = final_mastery - induction_trough
        recovery_fraction = (
            recovery_progress / recoverable_gap
            if recoverable_gap > 0.0
            else None
        )
        routing_sufficient = bool(
            induction_families
            and len(recovery_families) >= len(induction_families)
        )
        routed_families: set[str] = set()
        routing_sufficient_session = None
        presentations_until_routing_sufficient = None
        recovery_presentations = 0
        for session in sessions[first_recovered:]:
            recovery_presentations += session["attempted"]
            routed_families.update(
                step["family_id"]
                for step in session["trace"]
                if step["objective_id"] == target_objective_id
                and step["correct"]
            )
            if (
                routing_sufficient_session is None
                and induction_families
                and len(routed_families) >= len(induction_families)
            ):
                routing_sufficient_session = session["index"]
                presentations_until_routing_sufficient = (
                    recovery_presentations
                )
        recovery = {
            "first_recovery_session": first_recovered + 1,
            "baseline_mastery": baseline_mastery,
            "induction_trough_mastery": induction_trough,
            "pre_recovery_mastery": target_path[first_recovered - 1],
            "final_mastery": final_mastery,
            "change_from_trough": recovery_progress,
            "fraction_of_baseline_gap_recovered": recovery_fraction,
            "induction_families": induction_families,
            "recovery_families": recovery_families,
            "recovery_target_attempt_count": len(recovery_steps),
            "recovery_total_presentation_count": sum(
                session["attempted"]
                for session in sessions[first_recovered:]
            ),
            "recovery_target_attempt_fraction": (
                len(recovery_steps)
                / sum(
                    session["attempted"]
                    for session in sessions[first_recovered:]
                )
            ),
            "routing_sufficient": routing_sufficient,
            "routing_sufficient_session": routing_sufficient_session,
            "presentations_until_routing_sufficient": (
                presentations_until_routing_sufficient
            ),
            "recovery_observed": bool(
                routing_sufficient
                and recovery_fraction is not None
                and recovery_fraction >= 0.75
            ),
            "criterion": (
                "Route at least as many distinct correct target families after "
                "the behavior change as incorrect target families during "
                "induction, then recover at least 75% of the objective-mastery "
                "gap from the induction trough back toward its cold-start prior."
            ),
            "interpretation": (
                "This measures posterior responsiveness after an imposed "
                "behavior change, not teaching efficacy or causal learning."
            ),
        }
    return {
        "target_objective_id": target_objective_id,
        "schedule_regime": regime,
        "weak_schedule": list(weak_schedule),
        "sessions": sessions,
        "baseline_objective_snapshot": baseline_snapshot,
        "discovery": metrics,
        "recovery": recovery,
        "persistent_gap_episode_audit": episode_audit,
        "certificate_inflation_violations": certificate_violations,
        "integrity": {
            "ok": integrity["ok"],
            "event_count": integrity["event_count"],
            "stream_count": integrity["stream_count"],
        },
        "projection_replay": {
            "ok": replay["ok"],
            "source_projection_matches_replay": replay[
                "source_projection_matches_replay"
            ],
            "commitment_matches_replay": replay[
                "commitment_matches_replay"
            ],
        },
    }


def build_report(
    *,
    corpus: Path = DEFAULT_CORPUS,
    topic: str = DEFAULT_TOPIC,
    targets: Iterable[str] = DEFAULT_TARGETS,
    sessions: int = 3,
    steps_per_session: int = 10,
    seed: int = 431,
    include_recovery: bool = True,
) -> dict[str, Any]:
    selected_targets = tuple(dict.fromkeys(targets))
    if not selected_targets:
        raise ValueError("At least one target objective is required.")
    if sessions <= 0 or steps_per_session <= 0:
        raise ValueError("sessions and steps_per_session must be positive.")
    with tempfile.TemporaryDirectory(prefix="tsq-objective-discovery-") as raw:
        directory = Path(raw)
        cases = [
            run_case(
                database_path=directory / f"case-{index}.db",
                corpus=corpus,
                topic=topic,
                target_objective_id=target,
                weak_schedule=(True,) * sessions,
                steps_per_session=steps_per_session,
                seed=seed + 1_003 * index,
            )
            for index, target in enumerate(selected_targets)
        ]
        recovery = None
        if include_recovery:
            recovery_target = selected_targets[0]
            recovery = run_case(
                database_path=directory / "recovery.db",
                corpus=corpus,
                topic=topic,
                target_objective_id=recovery_target,
                weak_schedule=(True, True, *([False] * 7)),
                steps_per_session=max(6, steps_per_session),
                seed=seed + 99_991,
            )
        bounded_recovery = bounded_family_recovery_probe(
            corpus=corpus,
            objective_id=selected_targets[0],
        )

    deterministic = {
        "lab_version": LAB_VERSION,
        "corpus_sha256": corpus_hash(corpus),
        "topic": topic,
        "configuration": {
            "targets": list(selected_targets),
            "sessions": sessions,
            "steps_per_session": steps_per_session,
            "session_spacing_days": 45,
            "seed": seed,
            "include_recovery": include_recovery,
            "feedback_protocol": SIMULATION_FEEDBACK_PROTOCOL_VERSION,
        },
        "cases": cases,
        "recovery_case": recovery,
        "bounded_family_recovery_probe": bounded_recovery,
        "findings": {
            "cases_passing_discovery_hypothesis": sum(
                case["discovery"]["hypothesis_passed"] for case in cases
            ),
            "case_count": len(cases),
            "all_integrity_and_replay_checks_passed": all(
                case["integrity"]["ok"]
                and case["projection_replay"]["ok"]
                for case in [*cases, *([recovery] if recovery else [])]
            ),
            "all_persistent_gap_episode_contracts_passed": all(
                case["persistent_gap_episode_audit"][
                    "all_contracts_passed"
                ]
                for case in [*cases, *([recovery] if recovery else [])]
            ),
            "persistent_gap_episode_spend_count": sum(
                case["persistent_gap_episode_audit"]["spend_count"]
                for case in [*cases, *([recovery] if recovery else [])]
            ),
            "certificate_inflation_observed": any(
                case["certificate_inflation_violations"]
                for case in [*cases, *([recovery] if recovery else [])]
            ),
            "recovery_observed": (
                recovery["recovery"]["recovery_observed"]
                if recovery and recovery["recovery"]
                else None
            ),
            "bounded_family_recovery_observed": bounded_recovery[
                "all_families_recovered"
            ],
        },
        "interpretation_boundary": (
            "This is a deterministic identifiability falsification probe over "
            "production TSQ behavior. It is not human calibration, evidence of "
            "causal teaching benefit, or a claim that selected-response items "
            "measure productive implementation skill."
        ),
    }
    return {
        **deterministic,
        "artifact_sha256": canonical_hash(deterministic),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    result.add_argument("--topic", default=DEFAULT_TOPIC)
    result.add_argument("--target", action="append", dest="targets")
    result.add_argument("--sessions", type=int, default=3)
    result.add_argument("--steps", type=int, default=10)
    result.add_argument("--seed", type=int, default=431)
    result.add_argument("--no-recovery", action="store_true")
    result.add_argument("--fail-on-hypothesis", action="store_true")
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--stdout", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    report = build_report(
        corpus=arguments.corpus,
        topic=arguments.topic,
        targets=arguments.targets or DEFAULT_TARGETS,
        sessions=arguments.sessions,
        steps_per_session=arguments.steps,
        seed=arguments.seed,
        include_recovery=not arguments.no_recovery,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.stdout:
        sys.stdout.write(encoded)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
        print(
            json.dumps(
                {
                    "artifact_sha256": report["artifact_sha256"],
                    **report["findings"],
                    "output": str(arguments.output),
                },
                indent=2,
                sort_keys=True,
            )
        )
    if arguments.fail_on_hypothesis and (
        report["findings"]["cases_passing_discovery_hypothesis"]
        != report["findings"]["case_count"]
        or report["findings"]["recovery_observed"] is False
        or not report["findings"]["bounded_family_recovery_observed"]
        or not report["findings"][
            "all_persistent_gap_episode_contracts_passed"
        ]
        or report["findings"]["persistent_gap_episode_spend_count"] <= 0
        or report["findings"]["certificate_inflation_observed"]
    ):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
