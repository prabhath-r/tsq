#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0

"""Run deterministic synthetic learners against the real adaptive engine."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path

from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.simulation import (
    DEFAULT_SIMULATION_START,
    BehavioralSimulator,
    SyntheticLearner,
    assert_behavioral_invariants,
)
from tsq.store import Database


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def profile_named(name: str, seed: int) -> SyntheticLearner:
    if name == "strong":
        return SyntheticLearner(
            name,
            default_ability=0.92,
            slip_probability=0.02,
            guess_probability=0.0,
            seed=seed,
        )
    if name == "weak":
        return SyntheticLearner(
            name,
            default_ability=0.18,
            slip_probability=0.10,
            guess_probability=0.01,
            seed=seed,
        )
    if name == "always-wrong":
        return SyntheticLearner(
            name,
            default_ability=0.0,
            slip_probability=1.0,
            guess_probability=0.0,
            seed=seed,
        )
    return SyntheticLearner(
        "intermediate",
        default_ability=0.55,
        slip_probability=0.04,
        guess_probability=0.02,
        seed=seed,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Audit selection, remediation, repetition, coverage, and calibration "
            "with deterministic synthetic learners."
        )
    )
    result.add_argument(
        "--corpus",
        type=Path,
        default=PROJECT_ROOT / "corpus" / "ai_curriculum.json",
    )
    result.add_argument("--database", type=Path)
    result.add_argument("--root", default="c_ai_learning_systems")
    result.add_argument(
        "--profile",
        choices=("weak", "intermediate", "strong", "always-wrong"),
        default="intermediate",
    )
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--trials", type=int, default=1)
    result.add_argument("--steps", type=int, default=40)
    result.add_argument("--learner-prefix", default="behavioral-audit")
    result.add_argument(
        "--allow-blockers",
        action="store_true",
        help="exploratory mode: print invariant failures but exit successfully",
    )
    result.add_argument(
        "--summary-only",
        action="store_true",
        help="omit per-trial traces from cohort JSON output",
    )
    result.add_argument(
        "--start-at",
        type=datetime.fromisoformat,
        default=DEFAULT_SIMULATION_START,
        help="timezone-aware ISO-8601 timestamp",
    )
    return result


def _check_report(report, failures: list[str]) -> None:
    try:
        assert_behavioral_invariants(report)
    except AssertionError as exc:
        failures.append(str(exc))


def run(
    arguments: argparse.Namespace,
    database_path: Path,
    *,
    invariant_failures: list[str] | None = None,
) -> dict[str, object]:
    failures = invariant_failures if invariant_failures is not None else []
    database = Database(database_path)
    database.initialize()
    database.import_corpus(*read_and_parse(arguments.corpus))
    simulator = BehavioralSimulator(AdaptiveEngine(database))
    profile = profile_named(arguments.profile, arguments.seed)
    if arguments.trials == 1:
        report = simulator.run(
            profile,
            learner_id=f"{arguments.learner_prefix}-0",
            root_concept_id=arguments.root,
            policy_seed=arguments.seed,
            max_steps=arguments.steps,
            start_at=arguments.start_at,
        )
        _check_report(report, failures)
        return report.summary()

    cohort = simulator.evaluate(
        profile,
        learner_id_prefix=arguments.learner_prefix,
        root_concept_id=arguments.root,
        policy_seeds=range(arguments.seed, arguments.seed + arguments.trials),
        max_steps=arguments.steps,
        start_at=arguments.start_at,
    )
    for index, report in enumerate(cohort.trials):
        before = len(failures)
        _check_report(report, failures)
        if len(failures) > before:
            failures[-1] = f"trial {index}: {failures[-1]}"
    return {
        "cohort": cohort.summary(),
        "trials": [report.summary() for report in cohort.trials],
    }


def main() -> int:
    arguments = parser().parse_args()
    if arguments.trials <= 0 or arguments.steps <= 0:
        raise SystemExit("--trials and --steps must be positive")
    if arguments.start_at.tzinfo is None or arguments.start_at.utcoffset() is None:
        raise SystemExit("--start-at must include a UTC offset")
    invariant_failures: list[str] = []
    if arguments.database:
        output = run(
            arguments,
            arguments.database,
            invariant_failures=invariant_failures,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="tsq-behavior-") as directory:
            output = run(
                arguments,
                Path(directory) / "audit.db",
                invariant_failures=invariant_failures,
            )
    if invariant_failures:
        output = {**output, "audit_failures": invariant_failures}
    if arguments.summary_only and "cohort" in output:
        output = {
            "cohort": output["cohort"],
            **(
                {"audit_failures": output["audit_failures"]}
                if "audit_failures" in output
                else {}
            ),
        }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if arguments.allow_blockers or not invariant_failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
