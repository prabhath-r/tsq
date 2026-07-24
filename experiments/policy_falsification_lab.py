#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0

"""Try to falsify TSQ's adaptive-policy claims on disposable databases.

This laboratory does not implement a simulation-only selector.  Deterministic
answer policies provide behavior, while the production engine chooses every
question, opens and closes remediation, updates projections, and persists the
event history.

The experiment is an identifiability and routing stress test, not evidence of
human learning efficacy.  A configured claim is called ``falsified`` only when
the relevant target received enough independent exposure and still missed its
declared criterion.  Missing exposure is reported as ``inconclusive`` rather
than silently counted as success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tsq.corpus import read_and_parse  # noqa: E402
from tsq.engine import AdaptiveEngine  # noqa: E402
from tsq.inference import classify_response_for_model  # noqa: E402
from tsq.learner import LearnerModel  # noqa: E402
from tsq.models import MAX_REMEDIATION_DEPTH, Presentation, SessionPhase  # noqa: E402
from tsq.replay import ProjectionReplay  # noqa: E402
from tsq.simulation import (  # noqa: E402
    BehavioralSimulator,
    SIMULATION_FEEDBACK_PROTOCOL_VERSION,
    SyntheticAnswer,
)
from tsq.store import Database  # noqa: E402


LAB_VERSION = "policy-falsification-lab-v3"
DEFAULT_CORPUS = PROJECT_ROOT / "corpus" / "ai_curriculum.json"
DEFAULT_TOPIC = "t_transformers"
DEFAULT_START = datetime(2110, 1, 4, 9, 0, tzinfo=timezone.utc)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "experiments" / "results" / "policy_falsification_lab.json"
)
MAIN_PHASES = frozenset(
    {SessionPhase.LEARN, SessionPhase.DIAGNOSE, SessionPhase.REVIEW}
)
MIN_LOCALIZATION_FAMILIES = 2
MIN_LOCALIZATION_CONTROLS = 2
RESPONSE_MODES = frozenset(
    {
        "deliberate_correct",
        "slow_correct",
        "fast_correct",
        "confident_wrong",
        "uncertain_abstain",
        "wrong_main_correct_support",
    }
)


class FalsificationInvariantError(RuntimeError):
    """The laboratory itself violated an integrity or behavior contract."""


@dataclass(frozen=True, slots=True)
class SessionPattern:
    """One deterministic latent-behavior regime for one spaced session."""

    default_mode: str = "deliberate_correct"
    objective_modes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.default_mode not in RESPONSE_MODES:
            raise ValueError(f"Unknown default response mode: {self.default_mode}")
        keys = [objective_id for objective_id, _ in self.objective_modes]
        if any(not objective_id for objective_id in keys):
            raise ValueError("Objective response-mode keys cannot be empty.")
        if len(keys) != len(set(keys)):
            raise ValueError("An objective can have only one mode per session.")
        unknown = sorted(
            {
                mode
                for _, mode in self.objective_modes
                if mode not in RESPONSE_MODES
            }
        )
        if unknown:
            raise ValueError(f"Unknown objective response modes: {unknown}")

    def mode_for(self, objective_id: str | None) -> str:
        if objective_id is not None:
            for candidate_id, mode in self.objective_modes:
                if candidate_id == objective_id:
                    return mode
        return self.default_mode

    def terms(self) -> dict[str, Any]:
        return {
            "default_mode": self.default_mode,
            "objective_modes": dict(self.objective_modes),
        }


@dataclass(frozen=True, slots=True)
class ProfileSpec:
    """A falsifiable longitudinal profile and its declared expectations."""

    id: str
    hypothesis: str
    sessions: tuple[SessionPattern, ...]
    target_objective_ids: tuple[str, ...] = ()
    claim_kind: str = "localization"
    recovery_boundary: int | None = None
    require_independent_verification: bool = False
    require_exploration: bool = False
    minimum_steps_per_session: int = 1

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Profile ID cannot be empty.")
        if not self.sessions:
            raise ValueError(f"Profile {self.id} needs at least one session.")
        if self.claim_kind not in {
            "localization",
            "evidence_quality",
            "recovery",
            "breadth_control",
        }:
            raise ValueError(f"Unknown profile claim kind: {self.claim_kind}")
        if len(self.target_objective_ids) != len(set(self.target_objective_ids)):
            raise ValueError("Target objective IDs must be unique.")
        if self.claim_kind != "breadth_control" and not self.target_objective_ids:
            raise ValueError(f"Profile {self.id} needs a target objective.")
        if self.recovery_boundary is not None and not (
            0 < self.recovery_boundary < len(self.sessions)
        ):
            raise ValueError(
                "Recovery boundary must split the session schedule."
            )
        if self.minimum_steps_per_session <= 0:
            raise ValueError("Minimum steps per session must be positive.")

    def terms(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "hypothesis": self.hypothesis,
            "claim_kind": self.claim_kind,
            "target_objective_ids": list(self.target_objective_ids),
            "recovery_boundary": self.recovery_boundary,
            "require_independent_verification": (
                self.require_independent_verification
            ),
            "require_exploration": self.require_exploration,
            "minimum_steps_per_session": self.minimum_steps_per_session,
            "sessions": [session.terms() for session in self.sessions],
        }


@dataclass(frozen=True, slots=True)
class ObjectiveBehaviorLearner:
    """Provide deterministic actions without influencing question selection."""

    name: str
    pattern: SessionPattern
    rule: str = "objective-behavior-matrix"

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
        mode = self.pattern.mode_for(question.objective_id)
        if mode == "wrong_main_correct_support":
            mode = (
                "confident_wrong"
                if presentation.phase in MAIN_PHASES
                else "slow_correct"
            )
        if mode == "uncertain_abstain":
            return SyntheticAnswer(
                selected_option_id=None,
                correct=False,
                ground_truth_probability=0.50,
                confidence=0.20,
                response_ms=7_000,
                hint_count=0,
            )

        correct = mode in {
            "deliberate_correct",
            "slow_correct",
            "fast_correct",
        }
        if correct:
            selected = question.correct_option
        else:
            distractors = sorted(
                (option for option in question.options if not option.correct),
                key=lambda option: option.id,
            )
            diagnostic = [
                option
                for option in distractors
                if option.diagnostic_objective_id == question.objective_id
                and option.misconception_id
            ]
            named = [
                option for option in distractors if option.misconception_id
            ]
            candidates = diagnostic or named
            if not candidates:
                raise FalsificationInvariantError(
                    f"Question {question.id} has no named misconception distractor."
                )
            selected = candidates[0]

        response_ms = {
            "deliberate_correct": 6_000,
            "slow_correct": 45_000,
            "fast_correct": 100,
            "confident_wrong": 6_000,
        }[mode]
        return SyntheticAnswer(
            selected_option_id=selected.id,
            correct=correct,
            ground_truth_probability=0.98 if correct else 0.02,
            confidence=0.95,
            response_ms=response_ms,
            hint_count=0,
        )


def default_profile_specs() -> tuple[ProfileSpec, ...]:
    """Return the fixed, reviewable methodology matrix."""

    repair = SessionPattern(
        objective_modes=(
            ("lo_causal_visibility", "wrong_main_correct_support"),
        )
    )
    mixed = SessionPattern(
        default_mode="slow_correct",
        objective_modes=(
            ("lo_attention_logit_scaling", "confident_wrong"),
            ("lo_incremental_kv_cache", "uncertain_abstain"),
        ),
    )
    fast = SessionPattern(
        default_mode="slow_correct",
        objective_modes=(
            ("lo_attention_permutation_order", "fast_correct"),
        ),
    )
    recovery_weak = SessionPattern(
        objective_modes=(
            ("lo_attention_logit_scaling", "confident_wrong"),
        )
    )
    recovery_strong = SessionPattern(
        objective_modes=(
            ("lo_attention_logit_scaling", "slow_correct"),
        )
    )
    breadth = SessionPattern(default_mode="slow_correct")
    return (
        ProfileSpec(
            id="localized_repair",
            hypothesis=(
                "One wrong main-phase distinction should become the lowest "
                "localized objective, receive bounded support, and require a "
                "fresh-family verification before the episode resolves."
            ),
            sessions=(repair,) * 3,
            target_objective_ids=("lo_causal_visibility",),
            require_independent_verification=True,
        ),
        ProfileSpec(
            id="mixed_dual_gap",
            hypothesis=(
                "A confident misconception and an abstained objective should "
                "both remain distinguishable from deliberately correct, slow "
                "objectives without trapping the learner in either gap."
            ),
            # The shorter three-by-five schedule exposed both targets but left
            # every deliberately strong control with only one family.  Six
            # spaced ten-question sessions make the declared comparison
            # symmetric instead of counting an underpowered result as success.
            sessions=(mixed,) * 6,
            target_objective_ids=(
                "lo_attention_logit_scaling",
                "lo_incremental_kv_cache",
            ),
            minimum_steps_per_session=10,
        ),
        ProfileSpec(
            id="fast_evidence_control",
            hypothesis=(
                "Implausibly fast correct answers should remain less certified "
                "than slow, credible correct answers and should not masquerade "
                "as independent retrieval."
            ),
            sessions=(fast,) * 2,
            target_objective_ids=("lo_attention_permutation_order",),
            claim_kind="evidence_quality",
        ),
        ProfileSpec(
            id="recovery_switch",
            hypothesis=(
                "After two spaced weak sessions, repeated credible correct "
                "answers from fresh target families should reverse most of the "
                "localized posterior gap without erasing the earlier evidence."
            ),
            sessions=(recovery_weak,) * 2 + (recovery_strong,) * 4,
            target_objective_ids=("lo_attention_logit_scaling",),
            claim_kind="recovery",
            recovery_boundary=2,
            minimum_steps_per_session=6,
        ),
        ProfileSpec(
            id="strong_breadth_control",
            hypothesis=(
                "A consistently strong learner should receive broad objective "
                "coverage plus explicitly gated related-topic exploration, not "
                "a narrow repeated path."
            ),
            sessions=(breadth,) * 2,
            claim_kind="breadth_control",
            require_exploration=True,
        ),
    )


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def localization_metrics(
    snapshots: Sequence[Mapping[str, Mapping[str, Any]]],
    *,
    baseline: Mapping[str, Mapping[str, Any]],
    target_objective_ids: Sequence[str],
) -> dict[str, Any]:
    """Measure ranking and separation without hiding missing exposure."""

    if not snapshots:
        raise ValueError("At least one objective snapshot is required.")
    final = snapshots[-1]
    targets = tuple(target_objective_ids)
    missing = sorted(set(targets) - set(final))
    if missing:
        raise FalsificationInvariantError(
            f"Target objectives are missing from the topic profile: {missing}"
        )
    target_rows = [final[objective_id] for objective_id in targets]
    observed_control_ids = [
        objective_id
        for objective_id in sorted(final)
        if objective_id not in targets
        and final[objective_id]["observed_response_families"] > 0
    ]
    observed_strong_ids = [
        objective_id
        for objective_id in observed_control_ids
        if final[objective_id]["observed_response_families"]
        >= MIN_LOCALIZATION_FAMILIES
    ]
    underpowered_control_ids = sorted(
        set(observed_control_ids) - set(observed_strong_ids)
    )
    # Rank only among objectives with symmetric independent-family support.
    # Cold objectives and one-family controls are not evidence that a declared
    # target should separate within a short horizon.
    ordered = sorted(
        (*targets, *observed_strong_ids),
        key=lambda objective_id: (
            final[objective_id]["mastery_probability"],
            objective_id,
        ),
    )
    observed_strong = [
        final[objective_id]["mastery_probability"]
        for objective_id in observed_strong_ids
    ]
    target_masteries = [
        row["mastery_probability"] for row in target_rows
    ]
    target_ranks = {
        objective_id: ordered.index(objective_id) + 1
        for objective_id in targets
    }
    lowest_set = set(ordered[: len(targets)])
    top_k_recall = len(lowest_set.intersection(targets)) / len(targets)
    tolerant_lowest_set = set(ordered[: len(targets) + 1])
    top_k_plus_one_recall = (
        len(tolerant_lowest_set.intersection(targets)) / len(targets)
    )
    pairwise = [
        target_mastery < final[strong_id]["mastery_probability"]
        for target_mastery in target_masteries
        for strong_id in observed_strong_ids
    ]
    false_strong_regressions = [
        {
            "objective_id": objective_id,
            "baseline_mastery": baseline[objective_id][
                "mastery_probability"
            ],
            "final_mastery": final[objective_id]["mastery_probability"],
        }
        for objective_id in observed_strong_ids
        if objective_id in baseline
        and final[objective_id]["mastery_probability"] + 1e-8
        < baseline[objective_id]["mastery_probability"]
    ]
    minimum_target_families = min(
        row["observed_response_families"] for row in target_rows
    )
    sufficient_exposure = bool(
        minimum_target_families >= MIN_LOCALIZATION_FAMILIES
        and len(observed_strong_ids) >= MIN_LOCALIZATION_CONTROLS
    )
    median_separation = (
        statistics.median(observed_strong)
        - statistics.median(target_masteries)
        if observed_strong
        else None
    )
    strict_separation = (
        min(observed_strong) - max(target_masteries)
        if observed_strong
        else None
    )
    passed = bool(
        sufficient_exposure
        and top_k_plus_one_recall == 1.0
        and median_separation is not None
        and median_separation >= 0.05
        and not false_strong_regressions
    )
    return {
        "target_objective_ids": list(targets),
        "target_ranks_lowest_first": target_ranks,
        "target_mastery_paths": {
            objective_id: [
                snapshot[objective_id]["mastery_probability"]
                for snapshot in snapshots
            ]
            for objective_id in targets
        },
        "minimum_target_observed_families": minimum_target_families,
        "observed_strong_objective_ids": observed_strong_ids,
        "underpowered_control_objective_ids": underpowered_control_ids,
        "top_k_recall": top_k_recall,
        "top_k_plus_one_recall": top_k_plus_one_recall,
        "pairwise_weak_below_strong_rate": (
            sum(pairwise) / len(pairwise) if pairwise else None
        ),
        "median_strong_minus_target_mastery": median_separation,
        "minimum_strong_minus_maximum_target_mastery": strict_separation,
        "false_strong_regressions": false_strong_regressions,
        "sufficient_exposure": sufficient_exposure,
        "criterion_passed": passed,
        "criterion": (
            "Every target and every control admitted to the strong comparison "
            "has at least two independent observed families; at least two such "
            "controls exist; all targets occupy the lowest k+1 comparable ranks "
            "(one tie/noise tolerance); median sufficiently observed strong "
            "mastery exceeds median target mastery by at least 0.05; and no "
            "sufficiently observed declared-strong objective falls below its "
            "cold-start prior. Cold and one-family controls are reported but "
            "cannot make a short-horizon run conclusive. Exact top-k recall is "
            "still reported."
        ),
    }


def _longest_run(values: Iterable[str | None]) -> int:
    longest = 0
    previous: str | None = None
    current = 0
    for value in values:
        if value is not None and value == previous:
            current += 1
        elif value is None:
            current = 0
        else:
            previous = value
            current = 1
        longest = max(longest, current)
    return longest


def _credible_exploration_support(step: Mapping[str, Any]) -> bool:
    if (
        not step["correct"]
        or step["pedagogical_role"] not in {"main", "exploration_probe"}
    ):
        return False
    response_class = classify_response_for_model(
        model_version=LearnerModel().model_version,
        correct=True,
        selected_option_id=step["selected_option_id"],
        selected_misconception_id=None,
        confidence=step["confidence"],
        response_ms=step["response_ms"],
        hint_count=step["hint_count"],
    )
    return bool(
        response_class.certifies_retrieval
        and (step["confidence"] is None or step["confidence"] >= 0.65)
    )


def routing_metrics(
    sessions: Sequence[Mapping[str, Any]],
    *,
    root_objective_ids: set[str],
    target_objective_ids: set[str],
) -> dict[str, Any]:
    traces = [session["trace"] for session in sessions]
    all_steps = [step for trace in traces for step in trace]
    main_steps = [
        step for step in all_steps if step["phase_before"] in {
            phase.value for phase in MAIN_PHASES
        }
    ]
    root_main = [
        step
        for step in main_steps
        if step["objective_id"] in root_objective_ids
    ]
    observed_root = sorted(
        {
            step["objective_id"]
            for step in all_steps
            if step["objective_id"] in root_objective_ids
        }
    )
    non_target_main = [
        step
        for step in root_main
        if step["objective_id"] not in target_objective_ids
    ]
    exploration_violations: list[dict[str, Any]] = []
    exploration_count = 0
    for session_index, trace in enumerate(traces, start=1):
        for index, step in enumerate(trace):
            if step["pedagogical_role"] != "exploration_probe":
                continue
            exploration_count += 1
            prior = trace[max(0, index - 3) : index]
            reasons: list[str] = []
            if step["phase_before"] not in {
                phase.value for phase in MAIN_PHASES
            }:
                reasons.append("outside_main_phase")
            if step["focus_objective_before"] is not None:
                reasons.append("active_objective_focus")
            if index < 3 or (index - 3) % 5 != 0:
                reasons.append("outside_bounded_schedule")
            if len(prior) != 3 or not all(
                _credible_exploration_support(candidate)
                for candidate in prior
            ):
                reasons.append("without_three_credible_successes")
            if reasons:
                exploration_violations.append(
                    {
                        "session": session_index,
                        "step": step["index"],
                        "reasons": reasons,
                    }
                )
    session_non_target_counts = [
        sum(
            step["phase_before"] in {
                phase.value for phase in MAIN_PHASES
            }
            and step["objective_id"] in root_objective_ids
            and step["objective_id"] not in target_objective_ids
            for step in trace
        )
        for trace in traces
    ]
    return {
        "root_objective_count": len(root_objective_ids),
        "observed_root_objective_ids": observed_root,
        "root_objective_coverage_fraction": (
            len(observed_root) / len(root_objective_ids)
            if root_objective_ids
            else 0.0
        ),
        "main_phase_root_presentations": len(root_main),
        "non_target_main_presentations": len(non_target_main),
        "non_target_main_fraction": (
            len(non_target_main) / len(root_main) if root_main else None
        ),
        "sessions_with_non_target_main": sum(
            count > 0 for count in session_non_target_counts
        ),
        "longest_main_phase_objective_run": _longest_run(
            step["objective_id"] for step in root_main
        ),
        "exploration_questions": exploration_count,
        "exploration_gate_violations": exploration_violations,
        "controlled_breadth_criterion": (
            "Observe at least 75% of root objectives, keep the longest "
            "main-phase same-objective run at or below three, and emit no "
            "exploration probe without the production gate's three credible "
            "successes, schedule position, main phase, and clear focus."
        ),
    }


def episode_metrics(sessions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    episodes = [
        episode
        for session in sessions
        for episode in session["episodes"]
    ]
    repeated = [
        episode
        for episode in episodes
        if episode["exact_repeat_count"] or episode["family_repeat_count"]
    ]
    overlong = [
        episode
        for episode in episodes
        if episode["longest_consecutive_failure_run"]
        > MAX_REMEDIATION_DEPTH
    ]
    resolved = [
        episode for episode in episodes if episode["outcome"] == "resolved"
    ]
    non_independent_resolutions = [
        episode
        for episode in resolved
        if not episode["independent_verification"]
    ]
    return {
        "episode_count": len(episodes),
        "outcomes": dict(
            sorted(Counter(episode["outcome"] for episode in episodes).items())
        ),
        "maximum_support_questions": max(
            (episode["support_question_count"] for episode in episodes),
            default=0,
        ),
        "maximum_consecutive_failure_run": max(
            (
                episode["longest_consecutive_failure_run"]
                for episode in episodes
            ),
            default=0,
        ),
        "repeat_violations": repeated,
        "remediation_bound_violations": overlong,
        "resolved_episode_count": len(resolved),
        "independently_verified_resolution_count": (
            len(resolved) - len(non_independent_resolutions)
        ),
        "non_independent_resolutions": non_independent_resolutions,
        "bound_passed": not repeated and not overlong,
        "verification_contract_passed": (
            bool(resolved) and not non_independent_resolutions
        ),
        "bound_criterion": (
            f"No remediation episode may repeat a question/family or exceed "
            f"{MAX_REMEDIATION_DEPTH} consecutive failed support checks."
        ),
        "verification_criterion": (
            "A resolved episode must contain a credible correct VERIFY-phase "
            "answer from a family distinct from its trigger and every earlier "
            "family in that episode."
        ),
    }


def recovery_metrics(
    *,
    sessions: Sequence[Mapping[str, Any]],
    snapshots: Sequence[Mapping[str, Mapping[str, Any]]],
    baseline: Mapping[str, Mapping[str, Any]],
    objective_id: str,
    boundary: int,
) -> dict[str, Any]:
    path = [
        snapshot[objective_id]["mastery_probability"]
        for snapshot in snapshots
    ]
    prior = baseline[objective_id]["mastery_probability"]
    trough = min(path[:boundary])
    final = path[-1]
    recoverable_gap = max(0.0, prior - trough)
    recovery_fraction = (
        (final - trough) / recoverable_gap
        if recoverable_gap > 0.0
        else None
    )
    induction = [
        step
        for session in sessions[:boundary]
        for step in session["trace"]
        if step["objective_id"] == objective_id and not step["correct"]
    ]
    recovery = [
        step
        for session in sessions[boundary:]
        for step in session["trace"]
        if step["objective_id"] == objective_id and step["correct"]
    ]
    induction_families = sorted({step["family_id"] for step in induction})
    recovery_families = sorted({step["family_id"] for step in recovery})
    routing_sufficient = bool(
        induction_families
        and len(recovery_families) >= len(induction_families)
    )
    sufficient_exposure = bool(
        len(induction_families) >= 2
        and len(recovery_families) >= 2
    )
    passed = bool(
        sufficient_exposure
        and routing_sufficient
        and recovery_fraction is not None
        and recovery_fraction >= 0.75
    )
    return {
        "objective_id": objective_id,
        "recovery_boundary_after_session": boundary,
        "mastery_path": path,
        "cold_start_mastery": prior,
        "induction_trough_mastery": trough,
        "final_mastery": final,
        "fraction_of_prior_gap_recovered": recovery_fraction,
        "induction_incorrect_families": induction_families,
        "recovery_correct_families": recovery_families,
        "routing_sufficient": routing_sufficient,
        "sufficient_exposure": sufficient_exposure,
        "criterion_passed": passed,
        "criterion": (
            "Observe at least two independent target families before and after "
            "the behavior switch; route at least as many distinct correct "
            "recovery families as distinct incorrect induction families; and "
            "recover at least 75% of the trough-to-cold-start mastery gap."
        ),
        "interpretation": (
            "This tests posterior reversibility under imposed answers. It does "
            "not show that remediation caused a human learner to improve."
        ),
    }


def _latest_session_id(database: Database, learner_id: str) -> str:
    with database.read() as connection:
        row = connection.execute(
            """SELECT id FROM sessions
               WHERE learner_id = ?
               ORDER BY created_at DESC, id DESC LIMIT 1""",
            (learner_id,),
        ).fetchone()
    if row is None:
        raise FalsificationInvariantError(
            f"No session exists for learner {learner_id}."
        )
    return str(row["id"])


def _serialize_session(
    report: Any,
    *,
    session_index: int,
    pattern: SessionPattern,
) -> dict[str, Any]:
    trace = [
        {
            "index": step.index,
            "question_id": step.question_id,
            "family_id": step.family_id,
            "objective_id": step.learning_objective_id,
            "correct": step.actual_correct,
            "selected_option_id": step.selected_option_id,
            "confidence": step.confidence,
            "response_ms": step.response_ms,
            "hint_count": step.hint_count,
            "phase_before": step.phase_before.value,
            "phase_after": step.phase_after.value,
            "pedagogical_role": step.pedagogical_role,
            "focus_objective_before": step.focus_objective_before,
            "focus_objective_after": step.focus_objective_after,
        }
        for step in report.steps
    ]
    episodes: list[dict[str, Any]] = []
    for episode in report.focus_episodes:
        support = trace[episode.start_step : episode.end_step + 1]
        consecutive = 0
        longest = 0
        for step in support:
            if (
                step["phase_before"]
                in {SessionPhase.REMEDIATE.value, SessionPhase.VERIFY.value}
                and not step["correct"]
            ):
                consecutive += 1
                longest = max(longest, consecutive)
            else:
                consecutive = 0
        verification_positions = [
            index
            for index, step in enumerate(support)
            if step["phase_before"] == SessionPhase.VERIFY.value
            and step["correct"]
            and _credible_exploration_support(
                {**step, "pedagogical_role": "main"}
            )
        ]
        independent_verification = False
        if verification_positions:
            verification_index = verification_positions[-1]
            verification_family = support[verification_index]["family_id"]
            earlier_families = {
                episode.trigger_family_id,
                *(
                    step["family_id"]
                    for step in support[:verification_index]
                ),
            }
            independent_verification = (
                verification_family not in earlier_families
                and episode.family_repeat_count == 0
            )
        episodes.append(
            {
                "session": session_index,
                "start_step": episode.start_step,
                "end_step": episode.end_step,
                "trigger_question_id": episode.trigger_question_id,
                "trigger_family_id": episode.trigger_family_id,
                "focus_objective_id": episode.initial_focus_objective_id,
                "support_question_count": episode.length,
                "family_ids": list(episode.family_ids),
                "outcome": episode.outcome,
                "exact_repeat_count": episode.exact_repeat_count,
                "family_repeat_count": episode.family_repeat_count,
                "longest_consecutive_failure_run": longest,
                "verification_step_positions": verification_positions,
                "independent_verification": independent_verification,
            }
        )
    return {
        "index": session_index,
        "pattern": pattern.terms(),
        "started_at": report.started_at.isoformat(),
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
        "trace": trace,
        "episodes": episodes,
        "behavior_signature": report.behavior_signature(),
    }


def _assessment(
    *,
    claim_id: str,
    sufficient: bool,
    passed: bool,
    criterion: str,
    measurements: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "assessment": (
            "supported_in_configured_cases"
            if sufficient and passed
            else ("falsified" if sufficient else "inconclusive")
        ),
        "sufficient_observation": sufficient,
        "criterion_passed": passed,
        "criterion": criterion,
        "measurements": dict(measurements),
    }


def run_profile(
    *,
    database_path: Path,
    corpus: Path,
    topic: str,
    spec: ProfileSpec,
    steps_per_session: int,
    seed: int,
) -> dict[str, Any]:
    if database_path.name == "tsq.db":
        raise ValueError("The falsification lab refuses to use a default database.")
    database = Database(database_path)
    database.initialize()
    database.import_corpus(*read_and_parse(corpus, include_catalog=True))
    engine = AdaptiveEngine(database)
    simulator = BehavioralSimulator(engine)
    learner_id = f"policy-falsification-{spec.id}"
    stable_name = f"policy-falsification:{spec.id}"
    engine.create_learner(learner_id, f"simulation:{stable_name}")
    baseline = objective_snapshot(
        engine, learner_id, topic=topic, now=DEFAULT_START
    )
    root_objective_ids = set(baseline)
    unknown_targets = sorted(set(spec.target_objective_ids) - root_objective_ids)
    if unknown_targets:
        raise FalsificationInvariantError(
            f"Profile {spec.id} targets objectives outside {topic}: "
            f"{unknown_targets}"
        )
    sessions: list[dict[str, Any]] = []
    snapshots: list[dict[str, dict[str, Any]]] = []
    effective_steps = max(
        steps_per_session, spec.minimum_steps_per_session
    )
    for index, pattern in enumerate(spec.sessions):
        started_at = DEFAULT_START + timedelta(days=45 * index)
        report = simulator.run(
            ObjectiveBehaviorLearner(name=stable_name, pattern=pattern),
            learner_id=learner_id,
            root_concept_id=topic,
            policy_seed=seed + 1_009 * index,
            max_steps=effective_steps,
            start_at=started_at,
            inter_item_delay=timedelta(minutes=4),
            trial_index=index,
            require_fresh_learner=index == 0,
            verify_idempotency=index == 0,
        )
        session_id = _latest_session_id(database, learner_id)
        engine.end_session(
            session_id,
            completed=True,
            reason="policy falsification laboratory boundary",
            idempotency_key=f"policy-falsification-end:{spec.id}:{index}",
            now=report.ended_at,
        )
        sessions.append(
            _serialize_session(
                report, session_index=index + 1, pattern=pattern
            )
        )
        snapshots.append(
            objective_snapshot(
                engine, learner_id, topic=topic, now=report.ended_at
            )
        )

    integrity = database.verify_integrity()
    replay = ProjectionReplay(database).check(learner_id)
    routing = routing_metrics(
        sessions,
        root_objective_ids=root_objective_ids,
        target_objective_ids=set(spec.target_objective_ids),
    )
    episodes = episode_metrics(sessions)
    localization = (
        localization_metrics(
            snapshots,
            baseline=baseline,
            target_objective_ids=spec.target_objective_ids,
        )
        if spec.target_objective_ids
        else None
    )
    recovery = (
        recovery_metrics(
            sessions=sessions,
            snapshots=snapshots,
            baseline=baseline,
            objective_id=spec.target_objective_ids[0],
            boundary=spec.recovery_boundary,
        )
        if spec.recovery_boundary is not None
        else None
    )
    target_steps = [
        step
        for session in sessions
        for step in session["trace"]
        if step["objective_id"] in spec.target_objective_ids
    ]
    fast_target = [step for step in target_steps if step["response_ms"] < 250]
    target_final = {
        objective_id: snapshots[-1][objective_id]
        for objective_id in spec.target_objective_ids
    }
    target_observed_families = sum(
        row["observed_response_families"] for row in target_final.values()
    )
    target_certified_families = sum(
        row["successful_retrieval_families"]
        for row in target_final.values()
    )
    strong_final = [
        row
        for objective_id, row in snapshots[-1].items()
        if objective_id not in spec.target_objective_ids
        and row["observed_response_families"] > 0
    ]
    strong_observed_families = sum(
        row["observed_response_families"] for row in strong_final
    )
    strong_certified_families = sum(
        row["successful_retrieval_families"] for row in strong_final
    )
    target_certification_rate = (
        target_certified_families / target_observed_families
        if target_observed_families
        else None
    )
    strong_certification_rate = (
        strong_certified_families / strong_observed_families
        if strong_observed_families
        else None
    )
    evidence_quality_passed = bool(
        fast_target
        and target_certification_rate == 0.0
        and strong_certification_rate is not None
        and strong_certification_rate - target_certification_rate >= 0.50
    )

    claims: list[dict[str, Any]] = []
    if spec.claim_kind == "localization" and localization is not None:
        claims.append(
            _assessment(
                claim_id=f"{spec.id}:localization",
                sufficient=localization["sufficient_exposure"],
                passed=localization["criterion_passed"],
                criterion=localization["criterion"],
                measurements={
                    "target_ranks_lowest_first": localization[
                        "target_ranks_lowest_first"
                    ],
                    "top_k_recall": localization["top_k_recall"],
                    "top_k_plus_one_recall": localization[
                        "top_k_plus_one_recall"
                    ],
                    "median_separation": localization[
                        "median_strong_minus_target_mastery"
                    ],
                    "pairwise_ordering_rate": localization[
                        "pairwise_weak_below_strong_rate"
                    ],
                },
            )
        )
    elif spec.claim_kind == "evidence_quality":
        claims.append(
            _assessment(
                claim_id=f"{spec.id}:timing_credibility",
                sufficient=bool(
                    fast_target
                    and target_observed_families
                    and strong_observed_families
                ),
                passed=evidence_quality_passed,
                criterion=(
                    "Observe at least one sub-250ms target answer, certify zero "
                    "target retrieval families, and leave the aggregate "
                    "slow-correct family certification rate at least 0.50 "
                    "above the fast-target certification rate."
                ),
                measurements={
                    "fast_target_answers": len(fast_target),
                    "target_successful_retrieval_families": {
                        objective_id: row[
                            "successful_retrieval_families"
                        ]
                        for objective_id, row in target_final.items()
                    },
                    "target_family_certification_rate": (
                        target_certification_rate
                    ),
                    "slow_strong_family_certification_rate": (
                        strong_certification_rate
                    ),
                    "median_strong_minus_target_mastery": (
                        localization[
                            "median_strong_minus_target_mastery"
                        ]
                        if localization
                        else None
                    ),
                },
            )
        )
    elif spec.claim_kind == "recovery" and recovery is not None:
        claims.append(
            _assessment(
                claim_id=f"{spec.id}:recovery",
                sufficient=recovery["sufficient_exposure"],
                passed=recovery["criterion_passed"],
                criterion=recovery["criterion"],
                measurements={
                    "fraction_of_prior_gap_recovered": recovery[
                        "fraction_of_prior_gap_recovered"
                    ],
                    "induction_family_count": len(
                        recovery["induction_incorrect_families"]
                    ),
                    "recovery_family_count": len(
                        recovery["recovery_correct_families"]
                    ),
                },
            )
        )
    elif spec.claim_kind == "breadth_control":
        breadth_passed = bool(
            routing["root_objective_coverage_fraction"] >= 0.75
            and routing["longest_main_phase_objective_run"] <= 3
            and not routing["exploration_gate_violations"]
            and (
                not spec.require_exploration
                or routing["exploration_questions"] > 0
            )
        )
        claims.append(
            _assessment(
                claim_id=f"{spec.id}:controlled_breadth",
                sufficient=bool(
                    sum(session["attempted"] for session in sessions) >= 8
                ),
                passed=breadth_passed,
                criterion=routing["controlled_breadth_criterion"],
                measurements={
                    "root_objective_coverage_fraction": routing[
                        "root_objective_coverage_fraction"
                    ],
                    "longest_main_phase_objective_run": routing[
                        "longest_main_phase_objective_run"
                    ],
                    "exploration_questions": routing[
                        "exploration_questions"
                    ],
                    "exploration_gate_violations": routing[
                        "exploration_gate_violations"
                    ],
                },
            )
        )

    claims.append(
        _assessment(
            claim_id=f"{spec.id}:bounded_remediation",
            sufficient=True,
            passed=episodes["bound_passed"],
            criterion=episodes["bound_criterion"],
            measurements={
                "episode_count": episodes["episode_count"],
                "maximum_consecutive_failure_run": episodes[
                    "maximum_consecutive_failure_run"
                ],
                "repeat_violation_count": len(
                    episodes["repeat_violations"]
                ),
            },
        )
    )
    if spec.target_objective_ids:
        controlled_routing_sufficient = bool(
            routing["main_phase_root_presentations"] >= 4
        )
        controlled_routing_passed = bool(
            controlled_routing_sufficient
            and routing["non_target_main_fraction"] is not None
            and routing["non_target_main_fraction"] >= 0.25
            and routing["longest_main_phase_objective_run"] <= 3
            and not routing["exploration_gate_violations"]
        )
        claims.append(
            _assessment(
                claim_id=f"{spec.id}:anti_tunneling",
                sufficient=controlled_routing_sufficient,
                passed=controlled_routing_passed,
                criterion=(
                    "Across at least four root-topic main presentations, retain "
                    "at least 25% non-target breadth, never exceed a three-item "
                    "same-objective main run, and emit no invalid exploration."
                ),
                measurements={
                    "main_phase_root_presentations": routing[
                        "main_phase_root_presentations"
                    ],
                    "non_target_main_fraction": routing[
                        "non_target_main_fraction"
                    ],
                    "longest_main_phase_objective_run": routing[
                        "longest_main_phase_objective_run"
                    ],
                    "exploration_gate_violation_count": len(
                        routing["exploration_gate_violations"]
                    ),
                },
            )
        )
    if spec.require_independent_verification:
        claims.append(
            _assessment(
                claim_id=f"{spec.id}:independent_verification",
                sufficient=episodes["resolved_episode_count"] > 0,
                passed=episodes["verification_contract_passed"],
                criterion=episodes["verification_criterion"],
                measurements={
                    "resolved_episode_count": episodes[
                        "resolved_episode_count"
                    ],
                    "independently_verified_resolution_count": episodes[
                        "independently_verified_resolution_count"
                    ],
                },
            )
        )
    all_gaps = [
        gap for session in sessions for gap in session["gaps"]
    ]
    all_steps = [
        step for session in sessions for step in session["trace"]
    ]
    response_patterns = {
        "correct": sum(step["correct"] for step in all_steps),
        "incorrect": sum(
            not step["correct"] and step["selected_option_id"] is not None
            for step in all_steps
        ),
        "uncertain_or_abstained": sum(
            step["selected_option_id"] is None for step in all_steps
        ),
        "fast_under_250ms": sum(
            step["response_ms"] < 250 for step in all_steps
        ),
        "slow_at_least_30s": sum(
            step["response_ms"] >= 30_000 for step in all_steps
        ),
    }
    infrastructure_ok = bool(integrity["ok"] and replay["ok"])
    return {
        "profile": spec.terms(),
        "effective_steps_per_session": effective_steps,
        "sessions": sessions,
        "baseline_objectives": baseline,
        "final_objectives": snapshots[-1],
        "localization": localization,
        "routing": routing,
        "remediation_and_verification": episodes,
        "recovery": recovery,
        "observed_response_patterns": response_patterns,
        "claims": claims,
        "blockers": all_gaps,
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
        "infrastructure_ok": infrastructure_ok,
    }


def build_report(
    *,
    corpus: Path = DEFAULT_CORPUS,
    topic: str = DEFAULT_TOPIC,
    specs: Sequence[ProfileSpec] | None = None,
    profile_ids: Iterable[str] | None = None,
    steps_per_session: int = 5,
    seed: int = 8_119,
) -> dict[str, Any]:
    if steps_per_session <= 0:
        raise ValueError("steps_per_session must be positive.")
    available = tuple(default_profile_specs() if specs is None else specs)
    if profile_ids is None:
        selected = available
    else:
        requested = tuple(dict.fromkeys(profile_ids))
        by_id = {spec.id: spec for spec in available}
        unknown = sorted(set(requested) - set(by_id))
        if unknown:
            raise ValueError(f"Unknown falsification profiles: {unknown}")
        selected = tuple(by_id[profile_id] for profile_id in requested)
    if not selected:
        raise ValueError("At least one falsification profile is required.")

    with tempfile.TemporaryDirectory(prefix="tsq-policy-falsification-") as raw:
        directory = Path(raw)
        profiles = [
            run_profile(
                database_path=directory / f"profile-{index}.db",
                corpus=corpus,
                topic=topic,
                spec=spec,
                steps_per_session=steps_per_session,
                seed=seed + 10_007 * index,
            )
            for index, spec in enumerate(selected)
        ]
    claims = [claim for profile in profiles for claim in profile["claims"]]
    deterministic = {
        "lab_version": LAB_VERSION,
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "topic": topic,
        "steps_per_session": steps_per_session,
        "seed": seed,
        "feedback_protocol": SIMULATION_FEEDBACK_PROTOCOL_VERSION,
        "profiles": profiles,
        "findings": {
            "profile_count": len(profiles),
            "claim_count": len(claims),
            "supported_claims": sum(
                claim["assessment"] == "supported_in_configured_cases"
                for claim in claims
            ),
            "falsified_claims": [
                claim["claim_id"]
                for claim in claims
                if claim["assessment"] == "falsified"
            ],
            "inconclusive_claims": [
                claim["claim_id"]
                for claim in claims
                if claim["assessment"] == "inconclusive"
            ],
            "all_integrity_and_replay_checks_passed": all(
                profile["infrastructure_ok"] for profile in profiles
            ),
            "profiles_with_blockers": [
                profile["profile"]["id"]
                for profile in profiles
                if profile["blockers"]
            ],
        },
        "limitations": [
            (
                "Deterministic answer policies are mechanism probes, not "
                "representative humans and not evidence of teaching efficacy."
            ),
            (
                "Objective labels are supplied to the generator. The learner "
                "model sees only authored item mappings and submitted behavior."
            ),
            (
                "Ranking can only identify distinctions the released corpus "
                "actually routes; missing target exposure is inconclusive."
            ),
            (
                "Timing and confidence are untrusted self-report telemetry. "
                "This lab checks conservative use, not authenticity."
            ),
            (
                "A finite profile matrix can falsify a declared criterion in "
                "these cases, but cannot prove universal optimality or fairness."
            ),
            (
                "Recovery is an imposed behavior switch and posterior "
                "reversibility check, not a causal estimate of remediation."
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
    result.add_argument("--topic", default=DEFAULT_TOPIC)
    result.add_argument(
        "--profile",
        dest="profiles",
        action="append",
        choices=[spec.id for spec in default_profile_specs()],
        help="Run one profile; repeat to select several.",
    )
    result.add_argument("--steps", type=int, default=5)
    result.add_argument("--seed", type=int, default=8_119)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--stdout", action="store_true")
    result.add_argument("--fail-on-falsification", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    report = build_report(
        corpus=arguments.corpus,
        topic=arguments.topic,
        profile_ids=arguments.profiles,
        steps_per_session=arguments.steps,
        seed=arguments.seed,
    )
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
    if not report["findings"]["all_integrity_and_replay_checks_passed"]:
        return 2
    if (
        arguments.fail_on_falsification
        and report["findings"]["falsified_claims"]
    ):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
