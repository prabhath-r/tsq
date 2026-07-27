#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0

"""Exercise TSQ's mixed MCQ-to-productive-probe path on disposable databases.

The lab answers a real corpus question incorrectly, publishes two reviewed
fixture tasks pinned to that session's exact release, verifies objective-aware
probe selection, records assisted semantic work, imports an authority-free
evaluation, and proves that all productive evidence remains replayable shadow
data.  It never executes an artifact or opens the configured/default database.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tsq.artifact_intake import prepare_file_checkpoint  # noqa: E402
from tsq.corpus import read_and_parse  # noqa: E402
from tsq.engine import AdaptiveEngine  # noqa: E402
from tsq.evidence import (  # noqa: E402
    ActionPhase,
    CriterionScale,
    EvaluationStatus,
    LearningTask,
    RubricCriterion,
    ScorerKind,
    TaskModality,
    canonical_digest,
)
from tsq.performance import (  # noqa: E402
    ImportedCriterionResult,
    ImportedEvaluation,
)
from tsq.performance_ledger import (  # noqa: E402
    PerformanceLedger,
    PerformanceTaskRelease,
    TaskReleaseReview,
)
from tsq.performance_selection import recommend_performance_tasks  # noqa: E402
from tsq.replay import ProjectionReplay  # noqa: E402
from tsq.store import Database, question_content_hash  # noqa: E402


LAB_VERSION = "productive-probe-lab-v2"
START = datetime(2115, 3, 4, 9, 0, tzinfo=timezone.utc)
DEFAULT_CORPUS = PROJECT_ROOT / "corpus" / "ai_curriculum.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "results" / "productive_probe_lab.json"
_D0 = "0" * 64
_D1 = "1" * 64
_D2 = "2" * 64


def _projection_boundary(database: Database, learner_id: str, session_id: str) -> dict[str, Any]:
    with database.read() as connection:
        return {
            "learner_revision": connection.execute(
                "SELECT revision FROM learners WHERE id=?", (learner_id,)
            ).fetchone()["revision"],
            "session_revision": connection.execute(
                "SELECT revision FROM sessions WHERE id=?", (session_id,)
            ).fetchone()["revision"],
            "learner_projection_hash": database.learner_projection_hash(
                learner_id, connection
            ),
        }


def _persisted_database_text(database: Database) -> str:
    """Render every persisted value so private fixture sentinels can be sought."""

    values: list[str] = []
    with database.read() as connection:
        table_names = [
            row["name"]
            for row in connection.execute(
                """SELECT name FROM sqlite_master
                   WHERE type='table' AND name NOT LIKE 'sqlite_%'
                   ORDER BY name"""
            ).fetchall()
        ]
        for table_name in table_names:
            quoted_name = '"' + table_name.replace('"', '""') + '"'
            for row in connection.execute(f"SELECT * FROM {quoted_name}"):
                values.extend(
                    str(value) for value in row if value is not None
                )
    return "\n".join(values)


def _source_manifest(
    database: Database, release_id: str, source_id: str
) -> tuple[tuple[str, str], ...]:
    with database.read() as connection:
        row = connection.execute(
            """SELECT source.content_hash
               FROM release_sources membership
               JOIN sources source ON source.id=membership.source_id
               WHERE membership.release_id=? AND membership.source_id=?""",
            (release_id, source_id),
        ).fetchone()
    if row is None:
        raise RuntimeError("Selected question source is absent from its release.")
    return ((source_id, row["content_hash"]),)


def _tasks_for_focus(
    *,
    objective_id: str,
    concept_id: str,
    misconception_id: str | None,
    source_manifests: tuple[tuple[str, str], ...],
    stimulus_digest: str,
) -> tuple[LearningTask, LearningTask]:
    suffix = objective_id.removeprefix("lo_")
    misconception_ids = (
        (misconception_id,) if misconception_id is not None else ()
    )
    exact = LearningTask(
        id=f"task_lab_{suffix}_debug",
        version=1,
        family_id=f"family_lab_{suffix}_debug",
        title="Diagnose the selected-response evidence boundary",
        modality=TaskModality.DEBUGGING,
        criteria=(
            RubricCriterion(
                id=f"criterion_lab_{suffix}",
                name="Objective-specific diagnosis",
                scale=CriterionScale.CONTINUOUS,
                concept_weights=((concept_id, 1.0),),
                objective_weights=((objective_id, 1.0),),
                dependence_group=f"dependence_lab_{suffix}",
                misconception_ids=misconception_ids,
                evidence_cap=0.8,
                dependence_cap=0.8,
            ),
        ),
        instructions=(
            "Inspect the pinned diagnostic stimulus, identify the violated "
            "invariant, and submit only a content-addressed repair artifact."
        ),
        source_manifests=source_manifests,
        administration_id="productive_probe_lab_admin_v1",
        administration_manifest_digest=_D0,
        stimulus_id=f"stimulus_lab_{suffix}",
        stimulus_digest=stimulus_digest,
    )
    alternative = LearningTask(
        id=f"task_lab_{suffix}_explain",
        version=1,
        family_id=f"family_lab_{suffix}_explain",
        title="Explain the broader concept boundary",
        modality=TaskModality.EXPLANATION,
        criteria=(
            RubricCriterion(
                id=f"criterion_lab_{suffix}_concept",
                name="Concept-level explanation",
                scale=CriterionScale.CONTINUOUS,
                concept_weights=((concept_id, 1.0),),
                dependence_group=f"dependence_lab_{suffix}_concept",
            ),
        ),
        instructions=(
            "Explain the pinned concept boundary and submit only the digest of "
            "the explanation artifact."
        ),
        source_manifests=source_manifests,
        administration_id="productive_probe_lab_admin_v1",
        administration_manifest_digest=_D0,
        stimulus_id=f"stimulus_lab_{suffix}_concept",
        stimulus_digest=_D2,
    )
    return exact, alternative


def run_once(database_path: Path, corpus_path: Path = DEFAULT_CORPUS) -> dict[str, Any]:
    database = Database(database_path)
    database.initialize()
    database.import_corpus(*read_and_parse(corpus_path, include_catalog=True))
    engine = AdaptiveEngine(database)
    learner_id = "productive-probe-lab"
    engine.create_learner(learner_id, "Productive Probe Lab")
    session = engine.start_session(
        learner_id,
        "t_transformers",
        seed=719,
        now=START,
    )
    presentation = engine.next_question(
        session["id"], now=START + timedelta(minutes=1)
    )
    distractors = sorted(
        (option for option in presentation.question.options if not option.correct),
        key=lambda option: (
            option.misconception_id is None,
            option.id,
        ),
    )
    selected = distractors[0]
    engine.submit_answer(
        presentation.decision_id,
        selected.id,
        confidence=0.95,
        response_ms=8_000,
        hint_count=0,
        idempotency_key="productive-lab-wrong-answer",
        now=START + timedelta(minutes=2),
    )
    focused = database.get_session(session["id"])
    objective_id = focused["focus_objective_id"]
    if objective_id is None:
        objective_id = presentation.question.objective_id
    if objective_id is None:
        raise RuntimeError("Transformer laboratory question has no objective binding.")
    objectives = {
        objective.id: objective
        for objective in database.get_learning_objectives(
            focused["corpus_release_id"]
        )
    }
    objective = objectives[objective_id]
    source_id = sorted(presentation.question.source_ids)[0]
    source_manifests = _source_manifest(
        database, focused["corpus_release_id"], source_id
    )
    exact_task, alternative_task = _tasks_for_focus(
        objective_id=objective_id,
        concept_id=objective.primary_concept_id,
        misconception_id=focused["focus_misconception_id"],
        source_manifests=source_manifests,
        stimulus_digest=canonical_digest(
            {
                "question_id": presentation.question.id,
                "question_version": presentation.question.version,
                "question_content_hash": question_content_hash(
                    presentation.question
                ),
            }
        ),
    )
    task_release = PerformanceTaskRelease(
        title="Reviewed productive-probe laboratory fixture",
        corpus_release_id=focused["corpus_release_id"],
        review=TaskReleaseReview(
            reviewer_kind="human",
            reviewer_id="independent_productive_lab_reviewer",
            reviewed_at=START.isoformat(),
            independent_of_author=True,
            attestation_digest=_D1,
        ),
        tasks=(("pilot", exact_task), ("pilot", alternative_task)),
    )
    ledger = PerformanceLedger(database)
    released = ledger.publish_release(
        task_release, now=START + timedelta(minutes=3)
    )

    before = _projection_boundary(database, learner_id, session["id"])
    first = recommend_performance_tasks(
        database, session["id"], limit=5, now=START + timedelta(minutes=4)
    )
    attempt = ledger.start_attempt(
        session["id"],
        first["recommendations"][0]["task_id"],
        task_version=first["recommendations"][0]["task_version"],
        task_release_id=first["recommendations"][0]["task_release_id"],
        idempotency_key="productive-lab-task-start",
        now=START + timedelta(minutes=5),
    )
    ledger.record_action(
        attempt["id"],
        "hint_requested",
        {"hint_id": "productive_lab_hint", "level": 1},
        phase="assisted",
        idempotency_key="productive-lab-hint",
        now=START + timedelta(minutes=6),
    )
    artifact_sentinels = (
        "TSQ_PRIVATE_DIAGNOSIS_719 future keys precede normalization",
        "TSQ_PRIVATE_REPAIR_719 preserve each inclusive causal prefix",
    )
    artifact_material = ("\n".join(artifact_sentinels) + "\n").encode("utf-8")
    expected_artifact_digest = hashlib.sha256(artifact_material).hexdigest()
    artifact_path = database_path.parent / "learner-repair-artifact.txt"
    artifact_path.write_bytes(artifact_material)
    artifact_checkpoint = prepare_file_checkpoint(
        artifact_path,
        kind="artifact",
        artifact_kind="causal_mask_repair_v1",
    )
    recorded_artifact = ledger.record_action(
        attempt["id"],
        artifact_checkpoint.action_kind.value,
        artifact_checkpoint.payload,
        phase="assisted",
        idempotency_key="productive-lab-artifact",
        now=START + timedelta(minutes=7),
    )
    ledger.record_action(
        attempt["id"],
        "check_run",
        {
            "check_set_id": "productive_lab_checks",
            "passed": 2,
            "failed": 1,
            "errored": 0,
            "skipped": 0,
            "result_digest": _D2,
        },
        phase="assisted",
        idempotency_key="productive-lab-check",
        now=START + timedelta(minutes=8),
    )
    submission_checkpoint = prepare_file_checkpoint(
        artifact_path,
        kind="submission",
    )
    submitted = ledger.record_action(
        attempt["id"],
        submission_checkpoint.action_kind.value,
        submission_checkpoint.payload,
        phase="assisted",
        idempotency_key="productive-lab-submit",
        now=START + timedelta(minutes=10),
    )
    criterion = exact_task.criteria[0]
    scored = ledger.import_evaluation(
        attempt["id"],
        ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id=criterion.id,
                    status=EvaluationStatus.VALID,
                    score=0.4,
                    outcome_code="assisted_partial_repair",
                    phase=ActionPhase.ASSISTED,
                    source_action_ids=(submitted["id"],),
                    misconception_ids=criterion.misconception_ids,
                    reliability=0.8,
                ),
            )
        ),
        provider_id="productive_lab_import",
        provider_version="v1",
        declared_kind=ScorerKind.DETERMINISTIC,
        idempotency_key="productive-lab-evaluation",
        now=START + timedelta(minutes=11),
    )
    after = _projection_boundary(database, learner_id, session["id"])
    second = recommend_performance_tasks(
        database, session["id"], limit=5, now=START + timedelta(minutes=12)
    )
    pending = engine.next_question(
        session["id"], now=START + timedelta(minutes=13)
    )
    blocked = recommend_performance_tasks(
        database, session["id"], limit=5, now=START + timedelta(minutes=14)
    )
    attempt_report = ledger.report(attempt["id"])
    session_report = engine.session_report(
        session["id"], now=START + timedelta(minutes=14)
    )
    replay = ProjectionReplay(database).check(learner_id)
    integrity = database.verify_integrity()
    persisted_database_text = _persisted_database_text(database)
    artifact_private_material_absent = all(
        private_value not in persisted_database_text
        for private_value in (
            artifact_path.name,
            str(artifact_path),
            *artifact_sentinels,
        )
    )

    normalized_reasons = sorted(
        {
            reason
            for record in scored["shadow_evidence"]["records"]
            for reason in record["reason_codes"]
        }
    )
    stable = {
        "question_id": presentation.question.id,
        "question_objective_id": presentation.question.objective_id,
        "selected_misconception_id": selected.misconception_id,
        "focus_objective_id": objective_id,
        "focus_misconception_id": focused["focus_misconception_id"],
        "first_task_ids": [item["task_id"] for item in first["recommendations"]],
        "first_scores": [item["score"] for item in first["recommendations"]],
        "second_task_ids": [item["task_id"] for item in second["recommendations"]],
        "family_constraint": second["fresh_family_constraint_applied"],
        "pending_blocker_codes": [
            item["code"]
            for item in blocked["selection_boundary"]["start_blockers"]
        ],
        "projection_unchanged": before == after,
        "shadow_weight": scored["shadow_evidence"]["total_evidence_weight"],
        "shadow_reason_codes": normalized_reasons,
        "attempt_status": attempt_report["status"],
        "action_count": attempt_report["action_count"],
        "artifact_checkpoint_digest": recorded_artifact["payload"][
            "artifact_digest"
        ],
        "submission_checkpoint_digest": submitted["payload"][
            "submission_digest"
        ],
        "artifact_digest_matches_submission": (
            recorded_artifact["payload"]["artifact_digest"]
            == submitted["payload"]["submission_digest"]
        ),
        "artifact_digest_matches_bytes": (
            recorded_artifact["payload"]["artifact_digest"]
            == expected_artifact_digest
            == submitted["payload"]["submission_digest"]
        ),
        "artifact_private_material_absent": (
            artifact_private_material_absent
        ),
        "artifact_executed": False,
        "session_productive_attempts": session_report[
            "productive_skill_shadow"
        ]["attempt_count"],
        "objective_binding_ids": session_report[
            "productive_skill_shadow"
        ]["scope_binding"]["objective_ids"],
        "pending_question_id": pending.question.id,
        "replay_ok": replay["ok"],
        "performance_projection_matches_replay": replay[
            "performance_projection_matches_replay"
        ],
        "integrity_ok": integrity["ok"],
    }
    failures: list[str] = []
    if not first["recommendations"] or first["recommendations"][0]["task_id"] != exact_task.id:
        failures.append("Exact active-objective task was not the first probe.")
    if exact_task.id in stable["second_task_ids"]:
        failures.append("Used productive family was recommended despite a fresh family.")
    if stable["pending_blocker_codes"] != ["pending_question"]:
        failures.append("Pending MCQ was not exposed as the exact start blocker.")
    if not stable["projection_unchanged"]:
        failures.append("Shadow productive activity changed a learner projection.")
    if stable["shadow_weight"] != 0.0:
        failures.append("Direct imported evaluation acquired evidence authority.")
    if not stable["artifact_digest_matches_submission"]:
        failures.append("Artifact and submission commitments diverged.")
    if not stable["artifact_digest_matches_bytes"]:
        failures.append("Artifact commitment did not match the fixture bytes.")
    if not stable["artifact_private_material_absent"]:
        failures.append("Private artifact material crossed the digest boundary.")
    if not stable["replay_ok"] or not stable["integrity_ok"]:
        failures.append("Event integrity or projection replay failed.")
    return {
        "lab_version": LAB_VERSION,
        "database": str(database_path),
        "task_release_id": released["release_id"],
        "stable_signature": stable,
        "stable_digest": canonical_digest(stable),
        "failures": failures,
        "ok": not failures,
    }


def run_lab(corpus_path: Path = DEFAULT_CORPUS) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="tsq-productive-probe-lab-") as directory:
        root = Path(directory)
        first = run_once(root / "first.db", corpus_path)
        second = run_once(root / "second.db", corpus_path)
    return {
        "lab_version": LAB_VERSION,
        "deterministic_rerun": first["stable_signature"] == second["stable_signature"],
        "stable_digest": first["stable_digest"],
        "stable_signature": first["stable_signature"],
        "failures": [*first["failures"], *second["failures"]],
        "ok": (
            first["ok"]
            and second["ok"]
            and first["stable_signature"] == second["stable_signature"]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args(argv)
    result = run_lab(args.corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.stdout:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "status": "ok" if result["ok"] else "failed",
                    "deterministic_rerun": result["deterministic_rerun"],
                    "stable_digest": result["stable_digest"],
                    "output": str(args.output),
                },
                sort_keys=True,
            )
        )
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
