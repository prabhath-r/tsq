#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0

"""Falsify one-step policy-shadow estimates on disposable TSQ databases.

Fresh synthetic learners contribute exactly one answered decision each.  The
production policy still selects every question and records its immutable v18
safe frontier.  The lab then uses the synthetic response generator's declared
probability for every frontier action as an oracle for three narrow, one-step
quantities: the live logging distribution, uniform safe-frontier selection,
and the frozen greedy challenger.

This is an estimator and overlap experiment, not evidence of human learning,
teaching benefit, retention, or a counterfactual adaptive trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tsq.corpus import read_and_parse  # noqa: E402
from tsq.engine import AdaptiveEngine  # noqa: E402
from tsq.policy import POLICY_VERSION  # noqa: E402
from tsq.policy_shadow import GREEDY_POLICY_VERSION  # noqa: E402
from tsq.policy_shadow_reporting import (  # noqa: E402
    MIN_EFFECTIVE_SAMPLE_RATIO,
    MIN_EFFECTIVE_SAMPLE_SIZE,
    POLICY_SHADOW_REPORT_VERSION,
    PROSPECTIVE_ONE_STEP_OPE_VERSION,
    build_policy_shadow_report,
)
from tsq.replay import ProjectionReplay  # noqa: E402
from tsq.simulation import SyntheticLearner  # noqa: E402
from tsq.store import Database  # noqa: E402


LAB_VERSION = "policy-shadow-comparison-lab-v1"
DEFAULT_CORPUS = PROJECT_ROOT / "corpus" / "ai_curriculum.json"
DEFAULT_TOPIC = "t_transformers"
DEFAULT_TRIALS_PER_PROFILE = 32
DEFAULT_SEED = 63_011
DEFAULT_START = datetime(2114, 2, 2, 9, 0, tzinfo=timezone.utc)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "experiments" / "results"
    / "policy_shadow_comparison_lab.json"
)
BOUND_ALPHA = 0.01
MIN_UNWEIGHTED_OBSERVATIONS = 30


@dataclass(frozen=True, slots=True)
class Profile:
    id: str
    learner: SyntheticLearner
    mode: str = "learn"

    def terms(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mode": self.mode,
            "response_model": self.learner.response_model,
            "default_ability": self.learner.default_ability,
            "default_objective_ability": (
                self.learner.default_objective_ability
            ),
            "objective_abilities": dict(self.learner.objective_abilities),
            "misconception_strengths": dict(
                self.learner.misconception_strengths
            ),
            "slip_probability": self.learner.slip_probability,
            "guess_probability": self.learner.guess_probability,
        }


def default_profiles() -> tuple[Profile, ...]:
    """Return smooth, misspecified, and localized response generators."""

    return (
        Profile(
            "weak_4pl",
            SyntheticLearner(
                "shadow-oracle:weak-4pl",
                default_ability=0.20,
                slip_probability=0.05,
                guess_probability=0.02,
                seed=101,
            ),
        ),
        Profile(
            "intermediate_4pl",
            SyntheticLearner(
                "shadow-oracle:intermediate-4pl",
                default_ability=0.55,
                slip_probability=0.04,
                guess_probability=0.02,
                seed=103,
            ),
            mode="diagnose",
        ),
        Profile(
            "strong_4pl",
            SyntheticLearner(
                "shadow-oracle:strong-4pl",
                default_ability=0.85,
                slip_probability=0.03,
                guess_probability=0.01,
                seed=107,
            ),
            mode="review",
        ),
        Profile(
            "threshold_misspecified",
            SyntheticLearner(
                "shadow-oracle:threshold",
                default_ability=0.55,
                response_model="discontinuous_threshold",
                slip_probability=0.03,
                guess_probability=0.01,
                seed=109,
            ),
            mode="diagnose",
        ),
        Profile(
            "ability_only",
            SyntheticLearner(
                "shadow-oracle:ability-only",
                default_ability=0.55,
                response_model="ability_only",
                slip_probability=0.03,
                guess_probability=0.01,
                seed=113,
            ),
            mode="review",
        ),
        Profile(
            "localized_weakness",
            SyntheticLearner(
                "shadow-oracle:localized",
                default_ability=0.82,
                objective_abilities={"lo_attention_value_routing": 0.16},
                misconception_strengths={"m_attention_is_hard_selection": 0.90},
                slip_probability=0.04,
                guess_probability=0.01,
                seed=127,
            ),
        ),
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def oracle_values(
    frontier: Sequence[Mapping[str, Any]],
    *,
    live_question_id: str,
    challenger_question_id: str,
    probabilities: Mapping[str, float],
) -> dict[str, float]:
    """Calculate exact declared synthetic values for one logged frontier."""

    if not frontier:
        raise ValueError("An oracle frontier cannot be empty.")
    question_ids = [item.get("question_id") for item in frontier]
    if any(
        type(question_id) is not str or not question_id
        for question_id in question_ids
    ):
        raise ValueError("Every oracle frontier entry needs a question ID.")
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Oracle frontier question IDs must be unique.")
    if live_question_id not in question_ids:
        raise ValueError("The live question is absent from the oracle frontier.")
    if challenger_question_id not in question_ids:
        raise ValueError("The challenger is absent from the oracle frontier.")
    missing = sorted(set(question_ids) - set(probabilities))
    if missing:
        raise ValueError(f"Oracle probabilities are missing questions: {missing}")
    logging_probabilities: list[float] = []
    for item in frontier:
        value = item.get("logging_probability")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            or float(value) > 1.0
        ):
            raise ValueError("Oracle logging probabilities must be in (0, 1].")
        logging_probabilities.append(float(value))
    if not math.isclose(
        sum(logging_probabilities), 1.0, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("Oracle logging probabilities must sum to one.")
    for question_id in question_ids:
        probability = probabilities[question_id]
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(float(probability))
            or not 0.0 <= float(probability) <= 1.0
        ):
            raise ValueError("Oracle response probabilities must be in [0, 1].")
    return {
        "behavior": sum(
            logging_probability * float(probabilities[question_id])
            for question_id, logging_probability in zip(
                question_ids, logging_probabilities, strict=True
            )
        ),
        "uniform": sum(float(probabilities[item]) for item in question_ids)
        / len(question_ids),
        "greedy": float(probabilities[challenger_question_id]),
        "live_action": float(probabilities[live_question_id]),
    }


def _bounded_error_limit(range_bounds: Sequence[float]) -> float | None:
    """A predeclared Hoeffding diagnostic for independent bounded trials."""

    if not range_bounds:
        return None
    count = len(range_bounds)
    squared_range_sum = sum(
        float(bound) ** 2 for bound in range_bounds
    )
    return math.sqrt(
        math.log(2.0 / BOUND_ALPHA)
        * squared_range_sum
        / (2.0 * count * count)
    )


def _assessment(
    *,
    estimate: float | None,
    oracle: float,
    range_bounds: Sequence[float],
    status: str,
) -> dict[str, Any]:
    limit = _bounded_error_limit(range_bounds)
    error = None if estimate is None else abs(float(estimate) - oracle)
    adequate = bool(
        status == "descriptive_only"
        and estimate is not None
        and limit is not None
    )
    return {
        "assessment": (
            "not_falsified_within_predeclared_bound"
            if adequate and error is not None and error <= limit
            else ("falsified" if adequate else "inconclusive")
        ),
        "adequate_information": adequate,
        "estimate": estimate,
        "oracle": oracle,
        "absolute_error": error,
        "predeclared_error_limit": limit,
        "bound_alpha": BOUND_ALPHA,
        "bound_scope": (
            "fresh independent synthetic one-step outcomes using the maximum "
            "target importance weight on each full frontier; not "
            "dependence-adjusted human inference"
        ),
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty sequence.")
    return sum(values) / len(values)


def _local_target_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: str,
) -> dict[str, Any]:
    if target not in {"uniform", "greedy"}:
        raise ValueError("Local target must be uniform or greedy.")
    weight_field = f"{target}_weight"
    range_field = f"{target}_range_bound"
    weights = [float(row[weight_field]) for row in rows]
    outcomes = [float(row["correct"]) for row in rows]
    weight_sum = sum(weights)
    squared_sum = sum(weight * weight for weight in weights)
    if not rows:
        status = "unavailable"
        reasons = ["no observations"]
        ess = 0.0
        ess_ratio = None
        ips = None
        snips = None
    elif weight_sum <= 0.0 or squared_sum <= 0.0:
        status = "unavailable"
        reasons = ["no target-action support"]
        ess = 0.0
        ess_ratio = 0.0
        ips = None
        snips = None
    else:
        ess = weight_sum * weight_sum / squared_sum
        ess_ratio = ess / len(rows)
        reasons = []
        if ess < MIN_EFFECTIVE_SAMPLE_SIZE:
            reasons.append("effective sample size is below 30")
        if ess_ratio < MIN_EFFECTIVE_SAMPLE_RATIO:
            reasons.append("effective sample ratio is below 0.10")
        status = "low_information" if reasons else "descriptive_only"
        weighted_reward = sum(
            weight * outcome
            for weight, outcome in zip(weights, outcomes, strict=True)
        )
        ips = weighted_reward / len(rows)
        snips = weighted_reward / weight_sum
    oracle = (
        _mean([float(row["oracle"][target]) for row in rows])
        if rows
        else None
    )
    return {
        "status": status,
        "low_information_reasons": reasons,
        "support_count": sum(weight > 0.0 for weight in weights),
        "support_rate": (
            sum(weight > 0.0 for weight in weights) / len(rows)
            if rows
            else None
        ),
        "weights": {
            "sum": weight_sum,
            "effective_sample_size": ess,
            "effective_sample_ratio": ess_ratio,
            "zero_count": sum(weight == 0.0 for weight in weights),
        },
        "ips": ips,
        "snips": snips,
        "oracle": oracle,
        "assessment": (
            None
            if oracle is None
            else _assessment(
                estimate=ips,
                oracle=oracle,
                range_bounds=[
                    float(row[range_field]) for row in rows
                ],
                status=status,
            )
        ),
    }


def _stratum_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    outcomes = [float(row["correct"]) for row in rows]
    behavior_oracle = (
        _mean([float(row["oracle"]["behavior"]) for row in rows])
        if rows
        else None
    )
    behavior_status = (
        "descriptive_only"
        if len(rows) >= MIN_UNWEIGHTED_OBSERVATIONS
        else ("low_information" if rows else "unavailable")
    )
    return {
        "observation_count": len(rows),
        "live_behavior": {
            "status": behavior_status,
            "observed_mean": _mean(outcomes) if rows else None,
            "oracle": behavior_oracle,
            "assessment": (
                None
                if behavior_oracle is None
                else _assessment(
                    estimate=_mean(outcomes),
                    oracle=behavior_oracle,
                    range_bounds=[1.0] * len(rows),
                    status=behavior_status,
                )
            ),
        },
        "uniform": _local_target_summary(rows, target="uniform"),
        "greedy": _local_target_summary(rows, target="greedy"),
    }


def _run_once(
    database_path: Path,
    *,
    corpus: Path,
    topic: str,
    profiles: Sequence[Profile],
    trials_per_profile: int,
    seed: int,
) -> dict[str, Any]:
    if database_path.name == "tsq.db":
        raise ValueError("The comparison lab refuses to use a default database.")
    parsed = read_and_parse(corpus, include_catalog=True)
    questions = parsed[4]
    question_by_id = {question.id: question for question in questions}
    database = Database(database_path)
    database.initialize()
    database.import_corpus(*parsed)
    engine = AdaptiveEngine(database)
    profile_by_learner: dict[str, Profile] = {}
    trial_count = 0
    for profile_index, profile in enumerate(profiles):
        for trial_index in range(trials_per_profile):
            learner_id = (
                f"shadow-oracle-{profile.id}-{trial_index:04d}"
            )
            profile_by_learner[learner_id] = profile
            engine.create_learner(
                learner_id, f"simulation:{profile.learner.name}"
            )
            policy_seed = seed + profile_index * 100_003 + trial_index * 1_009
            response_seed = seed + 50_000_017 + profile_index * 200_003 + trial_index
            started_at = DEFAULT_START + timedelta(seconds=trial_count * 20)
            session = engine.start_session(
                learner_id,
                topic_id=topic,
                mode=profile.mode,
                seed=policy_seed,
                now=started_at,
            )
            presentation = engine.next_question(
                session["id"], now=started_at
            )
            answer = profile.learner.answer(
                presentation,
                simulation_seed=response_seed,
                trial_index=trial_index,
                encounter=1,
            )
            answered_at = started_at + timedelta(
                milliseconds=answer.response_ms
            )
            engine.submit_answer(
                presentation.decision_id,
                answer.selected_option_id,
                confidence=answer.confidence,
                response_ms=answer.response_ms,
                hint_count=answer.hint_count,
                feedback_shown=False,
                idempotency_key=(
                    f"shadow-oracle-answer:{profile.id}:{trial_index}"
                ),
                now=answered_at,
            )
            engine.end_session(
                session["id"],
                completed=True,
                reason="one-step policy-shadow comparison boundary",
                idempotency_key=(
                    f"shadow-oracle-end:{profile.id}:{trial_index}"
                ),
                now=answered_at + timedelta(milliseconds=1),
            )
            trial_count += 1

    integrity = database.verify_integrity()
    source_before_analysis = database.path.read_bytes()
    report = build_policy_shadow_report(database)
    rows: list[dict[str, Any]] = []
    reader = Database(database.path, read_only=True)
    with reader.read() as connection:
        stored = connection.execute(
            """SELECT shadow.frontier_json,
                      shadow.live_question_id,
                      shadow.challenger_question_id,
                      shadow.agreement,
                      decision.learner_id,
                      decision.phase,
                      decision.propensity,
                      attempt.is_correct
               FROM policy_shadow_evaluations shadow
               JOIN decisions decision ON decision.id=shadow.decision_id
               JOIN attempts attempt ON attempt.decision_id=shadow.decision_id
               ORDER BY decision.created_at, decision.id"""
        ).fetchall()
    for row in stored:
        profile = profile_by_learner[row["learner_id"]]
        frontier = json.loads(row["frontier_json"])
        probabilities = {
            item["question_id"]: profile.learner.probability_correct(
                question_by_id[item["question_id"]]
            )
            for item in frontier
        }
        oracle = oracle_values(
            frontier,
            live_question_id=row["live_question_id"],
            challenger_question_id=row["challenger_question_id"],
            probabilities=probabilities,
        )
        propensity = float(row["propensity"])
        uniform_weight = (1.0 / len(frontier)) / propensity
        greedy_weight = (
            1.0 / propensity if bool(row["agreement"]) else 0.0
        )
        logging_by_question = {
            item["question_id"]: float(item["logging_probability"])
            for item in frontier
        }
        rows.append(
            {
                "profile": profile.id,
                "phase": row["phase"],
                "correct": int(row["is_correct"]),
                "oracle": oracle,
                "uniform_weight": uniform_weight,
                "greedy_weight": greedy_weight,
                "uniform_range_bound": max(
                    (1.0 / len(frontier)) / probability
                    for probability in logging_by_question.values()
                ),
                "greedy_range_bound": (
                    1.0
                    / logging_by_question[row["challenger_question_id"]]
                ),
            }
        )

    replay_learners: list[str] = []
    for profile in profiles:
        candidates = sorted(
            learner_id
            for learner_id, candidate in profile_by_learner.items()
            if candidate.id == profile.id
        )
        replay_learners.extend(candidates[:1])
        if len(candidates) > 1:
            replay_learners.extend(candidates[-1:])
    replay_results = [
        ProjectionReplay(database).check(learner_id)
        for learner_id in replay_learners
    ]
    source_after_analysis = database.path.read_bytes()

    count = len(rows)
    observed = [float(row["correct"]) for row in rows]
    uniform_weights = [float(row["uniform_weight"]) for row in rows]
    greedy_weights = [float(row["greedy_weight"]) for row in rows]
    uniform_range_bounds = [
        float(row["uniform_range_bound"]) for row in rows
    ]
    greedy_range_bounds = [
        float(row["greedy_range_bound"]) for row in rows
    ]
    behavior_oracle = _mean(
        [float(row["oracle"]["behavior"]) for row in rows]
    )
    uniform_oracle = _mean(
        [float(row["oracle"]["uniform"]) for row in rows]
    )
    greedy_oracle = _mean(
        [float(row["oracle"]["greedy"]) for row in rows]
    )
    behavior_estimate = _mean(observed)
    uniform_result = report["uniform_safe_frontier"]
    greedy_result = report["prospective_shadow"]["one_step_ope"]
    uniform_ips = uniform_result["raw_correctness"]["ips"]
    greedy_ips = greedy_result["raw_correctness"]["ips"]
    local_uniform_ips = sum(
        weight * outcome
        for weight, outcome in zip(
            uniform_weights, observed, strict=True
        )
    ) / count
    local_greedy_ips = sum(
        weight * outcome
        for weight, outcome in zip(
            greedy_weights, observed, strict=True
        )
    ) / count
    arithmetic_ok = bool(
        uniform_ips is not None
        and math.isclose(
            float(uniform_ips), local_uniform_ips, rel_tol=1e-12, abs_tol=1e-12
        )
        and (
            (
                greedy_ips is None
                and sum(greedy_weights) == 0.0
            )
            or (
                greedy_ips is not None
                and math.isclose(
                    float(greedy_ips),
                    local_greedy_ips,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
        )
    )
    behavior_status = (
        "descriptive_only"
        if count >= MIN_UNWEIGHTED_OBSERVATIONS
        else "low_information"
    )
    assessments = {
        "live_behavior": _assessment(
            estimate=behavior_estimate,
            oracle=behavior_oracle,
            range_bounds=[1.0] * count,
            status=behavior_status,
        ),
        "uniform_safe_frontier_ips": _assessment(
            estimate=uniform_ips,
            oracle=uniform_oracle,
            range_bounds=uniform_range_bounds,
            status=uniform_result["status"],
        ),
        "greedy_challenger_ips": _assessment(
            estimate=greedy_ips,
            oracle=greedy_oracle,
            range_bounds=greedy_range_bounds,
            status=greedy_result["status"],
        ),
    }
    profile_strata = {
        profile.id: _stratum_summary(
            [row for row in rows if row["profile"] == profile.id]
        )
        for profile in profiles
    }
    observed_phases = sorted({str(row["phase"]) for row in rows})
    phase_strata = {
        phase: _stratum_summary(
            [row for row in rows if row["phase"] == phase]
        )
        for phase in observed_phases
    }
    expected_phases = {profile.mode for profile in profiles}
    failures: list[str] = []
    if not integrity["ok"]:
        failures.append("Event or projection integrity failed.")
    if not arithmetic_ok:
        failures.append("Production and independently recomputed OPE differ.")
    if not all(result["ok"] for result in replay_results):
        failures.append("One or more sampled learner projections failed replay.")
    if source_before_analysis != source_after_analysis:
        failures.append("Read-only analysis mutated its source database.")
    if report["decision_counts"]["answered"] != count:
        failures.append("Answered decision count differs from oracle rows.")
    if report["prospective_shadow"]["evaluation_count"] != count:
        failures.append("Prospective shadow coverage differs from oracle rows.")
    if set(row["profile"] for row in rows) != {
        profile.id for profile in profiles
    }:
        failures.append("At least one declared profile was not observed.")
    if set(observed_phases) != expected_phases:
        failures.append("At least one declared session phase was not observed.")
    return {
        "lab_version": LAB_VERSION,
        "backend_versions": {
            "policy": POLICY_VERSION,
            "policy_shadow_report": POLICY_SHADOW_REPORT_VERSION,
            "prospective_ope": PROSPECTIVE_ONE_STEP_OPE_VERSION,
            "challenger": GREEDY_POLICY_VERSION,
        },
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "topic": topic,
        "trials_per_profile": trials_per_profile,
        "seed": seed,
        "profiles": [profile.terms() for profile in profiles],
        "observation_count": count,
        "profile_counts": {
            profile.id: sum(row["profile"] == profile.id for row in rows)
            for profile in profiles
        },
        "phase_counts": {
            phase: sum(row["phase"] == phase for row in rows)
            for phase in sorted({row["phase"] for row in rows})
        },
        "oracles": {
            "live_behavior": behavior_oracle,
            "uniform_safe_frontier": uniform_oracle,
            "greedy_challenger": greedy_oracle,
        },
        "reported": {
            "live_behavior_mean": behavior_estimate,
            "uniform": uniform_result,
            "greedy": greedy_result,
        },
        "assessments": assessments,
        "stratified": {
            "profiles": profile_strata,
            "phases": phase_strata,
            "boundary": (
                "Strata are descriptive overlap diagnostics. No multiplicity, "
                "dependence, or human-population adjustment is applied."
            ),
        },
        "invariants": {
            "integrity_ok": bool(integrity["ok"]),
            "production_arithmetic_matches": arithmetic_ok,
            "replay_checked_learner_count": len(replay_results),
            "replay_ok": all(result["ok"] for result in replay_results),
            "source_database_unchanged_by_analysis": (
                source_before_analysis == source_after_analysis
            ),
            "one_decision_per_fresh_learner": (
                count == len(profile_by_learner)
            ),
            "all_profiles_observed": (
                set(row["profile"] for row in rows)
                == {profile.id for profile in profiles}
            ),
            "all_declared_phases_observed": (
                set(observed_phases) == expected_phases
            ),
        },
        "findings": {
            "falsified_estimators": [
                name
                for name, assessment in assessments.items()
                if assessment["assessment"] == "falsified"
            ],
            "inconclusive_estimators": [
                name
                for name, assessment in assessments.items()
                if assessment["assessment"] == "inconclusive"
            ],
            "underpowered_greedy_profile_ids": [
                profile_id
                for profile_id, summary in profile_strata.items()
                if summary["greedy"]["status"] != "descriptive_only"
            ],
            "underpowered_greedy_phases": [
                phase
                for phase, summary in phase_strata.items()
                if summary["greedy"]["status"] != "descriptive_only"
            ],
        },
        "limitations": [
            (
                "Synthetic response probabilities are declared generator "
                "truth, not human calibration."
            ),
            (
                "Only one-step outcomes on behavior-policy-visited states "
                "are estimated."
            ),
            (
                "No alternate remediation, learner-state, retention, or "
                "transfer trajectory is run."
            ),
            (
                "A lower oracle response rate is not evidence of better "
                "teaching or diagnostic value."
            ),
            (
                "Importance-weight ESS measures concentration, not "
                "independent human sample size."
            ),
            (
                "Fully answered trials do not identify response-censoring "
                "or abandonment effects."
            ),
            (
                "Aggregate overlap does not establish profile- or phase-level "
                "support; every stratum is reported separately."
            ),
        ],
        "failures": failures,
        "ok": not failures,
    }


def run_lab(
    *,
    corpus: Path = DEFAULT_CORPUS,
    topic: str = DEFAULT_TOPIC,
    profiles: Sequence[Profile] | None = None,
    trials_per_profile: int = DEFAULT_TRIALS_PER_PROFILE,
    seed: int = DEFAULT_SEED,
    replicate: bool = True,
) -> dict[str, Any]:
    if type(trials_per_profile) is not int or trials_per_profile <= 0:
        raise ValueError("trials_per_profile must be a positive integer.")
    if type(seed) is not int:
        raise ValueError("seed must be an integer.")
    selected = tuple(default_profiles() if profiles is None else profiles)
    if not selected or len({profile.id for profile in selected}) != len(selected):
        raise ValueError("Profiles must be non-empty and have unique IDs.")
    if any(
        profile.mode not in {"learn", "diagnose", "review"}
        for profile in selected
    ):
        raise ValueError("Profile modes must be learn, diagnose, or review.")
    corpus = corpus.resolve()
    corpus_before = corpus.read_bytes()
    with tempfile.TemporaryDirectory(
        prefix="tsq-policy-shadow-comparison-"
    ) as directory:
        root = Path(directory)
        first = _run_once(
            root / "first.db",
            corpus=corpus,
            topic=topic,
            profiles=selected,
            trials_per_profile=trials_per_profile,
            seed=seed,
        )
        second = (
            _run_once(
                root / "second.db",
                corpus=corpus,
                topic=topic,
                profiles=selected,
                trials_per_profile=trials_per_profile,
                seed=seed,
            )
            if replicate
            else None
        )
    stable_first = {
        key: value
        for key, value in first.items()
        if key not in {"failures", "ok"}
    }
    stable_second = (
        None
        if second is None
        else {
            key: value
            for key, value in second.items()
            if key not in {"failures", "ok"}
        }
    )
    deterministic_rerun = (
        None if second is None else stable_first == stable_second
    )
    failures = list(first["failures"])
    if second is not None:
        failures.extend(second["failures"])
        if not deterministic_rerun:
            failures.append("Fresh-database rerun changed the stable artifact.")
    if corpus.read_bytes() != corpus_before:
        failures.append("The comparison lab mutated its source corpus.")
    signature = {
        **stable_first,
        "deterministic_rerun": deterministic_rerun,
    }
    return {
        "lab_version": LAB_VERSION,
        "deterministic_rerun": deterministic_rerun,
        "stable_digest": _digest(signature),
        "stable_signature": signature,
        "failures": failures,
        "ok": (
            first["ok"]
            and (second is None or second["ok"])
            and deterministic_rerun is not False
            and corpus.read_bytes() == corpus_before
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument(
        "--trials-per-profile",
        type=int,
        default=DEFAULT_TRIALS_PER_PROFILE,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-replicate", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stdout", action="store_true")
    parser.add_argument("--fail-on-falsification", action="store_true")
    arguments = parser.parse_args(argv)
    result = run_lab(
        corpus=arguments.corpus,
        topic=arguments.topic,
        trials_per_profile=arguments.trials_per_profile,
        seed=arguments.seed,
        replicate=not arguments.no_replicate,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if arguments.stdout:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "deterministic_rerun": result["deterministic_rerun"],
                    "findings": result["stable_signature"]["findings"],
                    "ok": result["ok"],
                    "output": str(arguments.output),
                    "stable_digest": result["stable_digest"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    if not result["ok"]:
        return 2
    if (
        arguments.fail_on_falsification
        and result["stable_signature"]["findings"]["falsified_estimators"]
    ):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
