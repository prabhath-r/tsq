#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0

"""Run TSQ's artifact boundary adversarially on disposable fresh databases.

This is an executable lab, not a unit test. It uses the real bundled catalog,
reviewed task/release APIs, file intake, artifact runner, ledger inspection,
integrity verifier, and projection-copy replay. Every result is shadow-only:
no learner mastery, skill authority, or certification is applied. The valid,
semantic-invalid, and malformed scenarios exercise process separation.
Timeout, start-failure, and post-admission crash outcomes use controlled
injections so their durable ledger paths are deterministic. The bundled runner
is not an OS, filesystem, or network sandbox.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tsq.artifact_intake import (  # noqa: E402
    capture_productive_artifact,
    prepare_file_checkpoint,
)
from tsq.artifact_runner import (  # noqa: E402
    CAUSAL_MASK_CHECK_SET_ID,
    ArtifactProcessReceipt,
    ArtifactResultCode,
    ArtifactRunOutcome,
    ArtifactRunResult,
    SyntheticArtifactRunnerRegistry,
    bundled_synthetic_binding,
)
from tsq.corpus import read_and_parse  # noqa: E402
from tsq.engine import AdaptiveEngine  # noqa: E402
from tsq.errors import ValidationError  # noqa: E402
from tsq.evidence import (  # noqa: E402
    ActionKind,
    CriterionScale,
    LearningTask,
    RubricCriterion,
    ScorerContract,
    ScorerKind,
    TaskModality,
    canonical_digest,
)
from tsq.performance_ledger import (  # noqa: E402
    PerformanceLedger,
    PerformanceTaskRelease,
    TaskReleaseReview,
)
from tsq.replay import ProjectionReplay  # noqa: E402
from tsq.store import Database  # noqa: E402


LAB_VERSION = "artifact-runner-lab-v1"
START = datetime(2121, 9, 10, 9, 0, tzinfo=timezone.utc)
CORPUS = ROOT / "corpus"
OUTPUT = ROOT / "experiments" / "results" / "artifact_runner_lab.json"
SCENARIOS: tuple[tuple[str, bytes], ...] = (
    (
        "valid",
        b'{  "schema_version" : 1,\n'
        b' "mask" : [[true, false], [true, true]]   }\n',
    ),
    (
        "semantic_invalid",
        b'{ "mask" : [[true, true], [true, true]],\n'
        b' "schema_version" : 1 }\n',
    ),
    (
        "malformed",
        b'{"schema_version":1,"mask":[[true]],'
        b'"private_marker":"MALFORMED_PRIVATE_731"\n',
    ),
    ("timeout", b'{\n "schema_version": 1, "mask": [[true]]\n}\n'),
    ("failure", b'{ "schema_version": 1,\n "mask": [[true]] }\n'),
    ("crash", b'{ "mask": [[true]], "schema_version": 1 }\n'),
)
EXPECTED = {
    "valid": ("completed", (2, 0, 0, 0)),
    "semantic_invalid": ("completed", (1, 1, 0, 0)),
    "malformed": ("invalid_artifact", (0, 0, 1, 0)),
    "timeout": ("timed_out", None),
    "failure": ("runner_failed", None),
    "crash": ("unresolved", None),
}
BOUNDARY = {
    "shadow_only": True,
    "artifact_content_persisted": False,
    "artifact_executed": False,
    "evaluation_created": False,
    "learner_projection_applied": False,
    "mastery_applied": False,
    "certification_applied": False,
    "skill_authority": False,
    "process_separated_contract": True,
    "operating_system_sandboxed": False,
    "filesystem_isolation_enforced": False,
    "network_isolation_enforced": False,
}


def projection(
    database: Database, learner_id: str, session_id: str
) -> tuple[int, int, str]:
    with database.read() as connection:
        return (
            connection.execute(
                "SELECT revision FROM learners WHERE id=?", (learner_id,)
            ).fetchone()["revision"],
            connection.execute(
                "SELECT revision FROM sessions WHERE id=?", (session_id,)
            ).fetchone()["revision"],
            database.learner_projection_hash(learner_id, connection),
        )


def persisted_bytes(database: Database) -> bytes:
    """Return logical values plus checkpointed DB/WAL bytes for leak probes."""

    chunks: list[bytes] = []
    with database.read() as connection:
        tables = connection.execute(
            """SELECT name FROM sqlite_master
               WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"""
        ).fetchall()
        for item in tables:
            table = '"' + item["name"].replace('"', '""') + '"'
            for row in connection.execute(f"SELECT * FROM {table}"):
                chunks.extend(
                    str(value).encode() for value in row if value is not None
                )
    connection = sqlite3.connect(database.path)
    try:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()
    for path in (
        database.path,
        Path(f"{database.path}-wal"),
        Path(f"{database.path}-shm"),
    ):
        if path.exists():
            chunks.append(path.read_bytes())
    return b"\n".join(chunks)


def publish_task(
    database: Database, binding, corpus_release_id: str
) -> tuple[PerformanceLedger, LearningTask, str]:
    with database.read() as connection:
        source = connection.execute(
            """SELECT source.id, source.content_hash
               FROM release_sources member
               JOIN sources source ON source.id=member.source_id
               WHERE member.release_id=?
                 AND source.id='src_vaswani_attention_2017'""",
            (corpus_release_id,),
        ).fetchone()
    if source is None:
        raise RuntimeError("Bundled Transformer source is unavailable.")
    contract = ScorerContract(
        kind=ScorerKind.DETERMINISTIC,
        scorer_id=binding.checker_id.value,
        scorer_version=binding.checker_version,
        authority_id="authority.synthetic.artifact-runner-lab",
        authority_manifest_digest="2" * 64,
        criterion_ids=("criterion_causal_mask_lab",),
        evidence_action_kinds=(
            ActionKind.ARTIFACT_CHECKPOINT,
            ActionKind.CHECK_RUN,
        ),
        check_set_manifests=(
            (binding.check_set_id, binding.check_set_manifest_digest),
        ),
        artifact_manifests=(
            (binding.artifact_kind, binding.artifact_manifest_digest),
        ),
    )
    task = LearningTask(
        id="task_artifact_runner_lab",
        version=1,
        family_id="family_artifact_runner_lab",
        title="Check a causal-mask matrix artifact",
        modality=TaskModality.DEBUGGING,
        criteria=(
            RubricCriterion(
                id="criterion_causal_mask_lab",
                name="Causal visibility",
                scale=CriterionScale.CONTINUOUS,
                concept_weights=(("c_causal_masking", 1.0),),
                dependence_group="artifact_runner_lab",
                evidence_cap=0.8,
                dependence_cap=0.8,
            ),
        ),
        instructions=(
            "Produce a JSON causal-mask matrix and run the pinned checker."
        ),
        source_manifests=((source["id"], source["content_hash"]),),
        administration_id="admin_artifact_runner_lab_v1",
        administration_manifest_digest="0" * 64,
        stimulus_id="stimulus_artifact_runner_lab_v1",
        stimulus_digest="1" * 64,
        scorer_contracts=(contract,),
    )
    release = PerformanceTaskRelease(
        title="Reviewed artifact-runner laboratory fixture",
        corpus_release_id=corpus_release_id,
        review=TaskReleaseReview(
            reviewer_kind="human",
            reviewer_id="independent_artifact_runner_lab_reviewer",
            reviewed_at=(START - timedelta(minutes=1)).isoformat(),
            independent_of_author=True,
            attestation_digest="2" * 64,
        ),
        tasks=(("pilot", task),),
    )
    ledger = PerformanceLedger(database)
    published = ledger.publish_release(release, now=START)
    return ledger, task, published["release_id"]


def injected_receipt(request, binding, scenario: str):
    outcome, code, started = {
        "timeout": (
            ArtifactRunOutcome.TIMED_OUT,
            ArtifactResultCode.WORKER_TIMEOUT,
            True,
        ),
        "failure": (
            ArtifactRunOutcome.WORKER_FAILED,
            ArtifactResultCode.WORKER_START_FAILED,
            False,
        ),
    }[scenario]
    return ArtifactProcessReceipt(
        request=request,
        binding=binding,
        result=ArtifactRunResult(
            checker_id=request.checker_id,
            checker_version=request.checker_version,
            artifact_sha256=request.artifact_sha256,
            outcome=outcome,
            outcome_codes=(code,),
            passed=0,
            failed=0,
            errored=1,
            skipped=0,
        ),
        worker_process_started=started,
    )


def run_scenario(
    *,
    database: Database,
    engine: AdaptiveEngine,
    ledger: PerformanceLedger,
    task: LearningTask,
    release_id: str,
    binding,
    root: Path,
    scenario: str,
    material: bytes,
    index: int,
) -> tuple[dict[str, Any], Path]:
    learner_id = f"artifact-runner-lab-{scenario}"
    engine.create_learner(learner_id, f"Artifact Lab {scenario}")
    now = START + timedelta(hours=index + 1)
    session = engine.start_session(
        learner_id,
        "t_transformers",
        seed=9100 + index,
        idempotency_key=f"artifact-lab-session-{scenario}",
        now=now,
    )
    attempt = ledger.start_attempt(
        session["id"],
        task.id,
        task_version=task.version,
        task_release_id=release_id,
        idempotency_key=f"artifact-lab-start-{scenario}",
        now=now + timedelta(minutes=1),
    )
    path = root / f"PRIVATE_ARTIFACT_PATH_{scenario}_731.json"
    path.write_bytes(material)
    checkpoint = prepare_file_checkpoint(
        path, kind="artifact", artifact_kind=binding.artifact_kind
    )
    snapshot = capture_productive_artifact(path)
    if checkpoint.sha256 != snapshot.sha256:
        raise RuntimeError("File checkpoint and captured snapshot diverged.")
    action = ledger.record_action(
        attempt["id"],
        checkpoint.action_kind.value,
        checkpoint.payload,
        idempotency_key=f"artifact-lab-checkpoint-{scenario}",
        now=now + timedelta(minutes=2),
    )
    before = projection(database, learner_id, session["id"])
    registry = SyntheticArtifactRunnerRegistry(allow_synthetic=True)
    registry.register(binding)
    real_run = registry.run
    calls = 0
    real_runner_path_used = False

    def invoke(request, artifact):
        nonlocal calls, real_runner_path_used
        calls += 1
        if scenario in {"valid", "semantic_invalid", "malformed"}:
            real_runner_path_used = True
            return real_run(request, artifact)
        if scenario in {"timeout", "failure"}:
            return injected_receipt(request, binding, scenario)
        raise RuntimeError("controlled response-channel crash")

    run = None
    crash_failed_closed = scenario != "crash"
    with patch.object(registry, "run", side_effect=invoke):
        try:
            run = ledger.run_artifact_check(
                attempt["id"],
                snapshot,
                registry,
                binding,
                check_set_id=CAUSAL_MASK_CHECK_SET_ID,
                artifact_action_id=action["id"],
                idempotency_key=f"artifact-lab-run-{scenario}",
                now=now + timedelta(minutes=3),
            )
        except ValidationError as exc:
            if scenario != "crash":
                raise
            crash_failed_closed = "remains unresolved" in str(exc)

    converged: bool | None = None
    no_retry: bool | None = None
    if scenario in {"valid", "crash"}:
        first_claim = None if run is None else run["claim_id"]
        with patch.object(
            registry,
            "run",
            side_effect=AssertionError("runner must not be reinvoked"),
        ) as blocked:
            replayed = ledger.run_artifact_check(
                attempt["id"],
                snapshot,
                registry,
                binding,
                check_set_id=CAUSAL_MASK_CHECK_SET_ID,
                artifact_action_id=action["id"],
                idempotency_key=f"artifact-lab-varied-{scenario}",
                now=now + timedelta(minutes=4),
            )
        no_retry = blocked.call_count == 0
        converged = replayed["idempotent_replay"] and (
            first_claim is None or first_claim == replayed["claim_id"]
        )
        run = replayed
    if run is None:
        raise RuntimeError("Crash scenario did not recover its durable claim.")

    listed = ledger.list_artifact_runs(attempt_id=attempt["id"])
    inspected = ledger.inspect_artifact_run(run["claim_id"])
    actions = ledger.list_actions(attempt["id"])
    with database.read() as connection:
        claim = connection.execute(
            "SELECT * FROM performance_artifact_run_claims WHERE id=?",
            (run["claim_id"],),
        ).fetchone()
        receipt = connection.execute(
            """SELECT * FROM performance_artifact_run_receipts
               WHERE claim_id=?""",
            (run["claim_id"],),
        ).fetchone()
        events = connection.execute(
            """SELECT event_type, session_id, metadata_json FROM events
               WHERE correlation_id=? ORDER BY stream_id, stream_version""",
            (attempt["id"],),
        ).fetchall()
    artifact_events = [
        event
        for event in events
        if event["event_type"].startswith("PerformanceArtifactRun")
    ]
    check = inspected["check_action"]
    result = (
        None
        if inspected["process_receipt"] is None
        else inspected["process_receipt"]["result"]
    )
    report = {
        "status": inspected["status"],
        "terminal": inspected["terminal"],
        "retry_allowed": inspected["retry_allowed"],
        "invocation_count": calls,
        "varied_key_converged": converged,
        "varied_key_no_retry": no_retry,
        "crash_failed_closed": crash_failed_closed,
        "controlled_injection": scenario in {"timeout", "failure", "crash"},
        "actual_process_invoked": real_runner_path_used,
        "execution_mode": (
            "real_child_process"
            if scenario in {"valid", "semantic_invalid", "malformed"}
            else (
                "controlled_receipt_injection"
                if scenario in {"timeout", "failure"}
                else "controlled_post_admission_exception"
            )
        ),
        "projection_unchanged": (
            before == projection(database, learner_id, session["id"])
        ),
        "claim_rows": len(listed),
        "claim": {
            "through_sequence": claim["through_sequence"],
            "artifact_digest": claim["artifact_digest"],
            "binding_digest": claim["binding_digest"],
        },
        "receipt": {
            "present": receipt is not None,
            "outcome": None if receipt is None else receipt["outcome"],
            "result_present": (
                receipt is not None and receipt["result_json"] is not None
            ),
            "check_action_present": (
                receipt is not None and receipt["check_action_id"] is not None
            ),
        },
        "actions": [item["action_type"] for item in actions],
        "check_counts": (
            None
            if check is None
            else {
                name: check["payload"][name]
                for name in ("passed", "failed", "errored", "skipped")
            }
        ),
        "events": [event["event_type"] for event in events],
        "artifact_events_shadow_only": all(
            json.loads(event["metadata_json"]).get("shadow_only") is True
            for event in artifact_events
        ),
        "observation_outside_session": all(
            event["session_id"] is None
            for event in artifact_events
            if event["event_type"] == "PerformanceArtifactRunObserved"
        ),
        "result": (
            None
            if result is None
            else {
                "outcome": result["outcome"],
                "codes": result["outcome_codes"],
                "semantic_digest": canonical_digest(result),
            }
        ),
        "ledger_boundary_flags_match_contract": all(
            inspected[name] is expected
            for name, expected in BOUNDARY.items()
            if name != "process_separated_contract"
        ),
    }
    return report, path


def run_once(database_path: Path, root: Path, corpus: Path) -> dict[str, Any]:
    database = Database(database_path)
    database.initialize()
    database.import_corpus(*read_and_parse(corpus, include_catalog=True))
    with database.read() as connection:
        release_id = connection.execute(
            "SELECT value FROM meta WHERE key='active_corpus_release'"
        ).fetchone()["value"]
        catalog = {
            "questions": connection.execute(
                "SELECT COUNT(*) FROM release_questions WHERE release_id=?",
                (release_id,),
            ).fetchone()[0],
            "topics": connection.execute(
                "SELECT COUNT(*) FROM release_topics WHERE release_id=?",
                (release_id,),
            ).fetchone()[0],
        }
    binding = bundled_synthetic_binding()
    ledger, task, task_release_id = publish_task(
        database, binding, release_id
    )
    engine = AdaptiveEngine(database)
    reports: dict[str, dict[str, Any]] = {}
    paths: list[Path] = []
    for index, (scenario, material) in enumerate(SCENARIOS):
        reports[scenario], path = run_scenario(
            database=database,
            engine=engine,
            ledger=ledger,
            task=task,
            release_id=task_release_id,
            binding=binding,
            root=root,
            scenario=scenario,
            material=material,
            index=index,
        )
        paths.append(path)

    integrity = database.verify_integrity()
    copy_path = root / "replayed.db"
    replay = ProjectionReplay(database).rebuild_copy(
        "artifact-runner-lab-valid", copy_path
    )
    copy = Database(copy_path, read_only=True)
    copy_integrity = copy.verify_integrity()
    with database.read() as connection:
        source_counts = tuple(
            connection.execute(
                """SELECT
                   (SELECT COUNT(*) FROM performance_artifact_run_claims),
                   (SELECT COUNT(*) FROM performance_artifact_run_receipts)"""
            ).fetchone()
        )
    with copy.read() as connection:
        copy_counts = tuple(
            connection.execute(
                """SELECT
                   (SELECT COUNT(*) FROM performance_artifact_run_claims),
                   (SELECT COUNT(*) FROM performance_artifact_run_receipts)"""
            ).fetchone()
        )
    persisted = persisted_bytes(database) + persisted_bytes(copy)
    raw_absent = all(material not in persisted for _, material in SCENARIOS)
    paths_absent = all(
        str(path).encode() not in persisted
        and path.name.encode() not in persisted
        for path in paths
    )
    boundary = {
        **BOUNDARY,
        "statement": (
            "Shadow-only observation; no learner mastery, skill authority, "
            "or certification is applied."
        ),
        "sandbox_statement": (
            "Process isolation is not an OS, filesystem, or network sandbox."
        ),
    }
    signature = {
        "lab_version": LAB_VERSION,
        "catalog": catalog,
        "task_id": task.id,
        "binding_digest": binding.digest,
        "scenarios": reports,
        "privacy": {
            "artifact_byte_sequences_absent": raw_absent,
            "artifact_paths_absent": paths_absent,
            "full_artifact_sequences_and_paths_absent_from_database_scan": (
                raw_absent and paths_absent
            ),
        },
        "integrity_and_copy_replay": {
            "source_integrity_ok": integrity["ok"],
            "copy_replay_ok": replay["ok"],
            "copy_integrity_ok": copy_integrity["ok"],
            "performance_projection_matches_replay": replay[
                "performance_projection_matches_replay"
            ],
            "artifact_run_counts_match": source_counts == copy_counts,
            "claims": source_counts[0],
            "receipts": source_counts[1],
        },
        "authority_boundary": boundary,
    }
    failures: list[str] = []
    for scenario, (expected_status, expected_counts) in EXPECTED.items():
        item = reports[scenario]
        counts = (
            None
            if item["check_counts"] is None
            else tuple(
                item["check_counts"][name]
                for name in ("passed", "failed", "errored", "skipped")
            )
        )
        if item["status"] != expected_status or counts != expected_counts:
            failures.append(f"{scenario}: outcome/check counts diverged")
        if item["invocation_count"] != 1:
            failures.append(f"{scenario}: runner invoked more than once")
        if (
            not item["projection_unchanged"]
            or not item["ledger_boundary_flags_match_contract"]
            or not item["artifact_events_shadow_only"]
            or not item["observation_outside_session"]
        ):
            failures.append(f"{scenario}: shadow authority boundary crossed")
    for scenario in ("valid", "crash"):
        if (
            not reports[scenario]["varied_key_converged"]
            or not reports[scenario]["varied_key_no_retry"]
        ):
            failures.append(f"{scenario}: varied key caused a retry")
    for scenario in ("valid", "semantic_invalid", "malformed"):
        if not reports[scenario]["actual_process_invoked"]:
            failures.append(f"{scenario}: real child-process path not exercised")
    for scenario in ("timeout", "failure", "crash"):
        if reports[scenario]["actual_process_invoked"]:
            failures.append(f"{scenario}: controlled injection invoked a process")
    if not reports["crash"]["crash_failed_closed"]:
        failures.append("crash did not preserve an unresolved claim")
    if not raw_absent or not paths_absent:
        failures.append("artifact bytes or paths crossed persistence")
    if (
        not integrity["ok"]
        or not replay["ok"]
        or not copy_integrity["ok"]
        or not replay["performance_projection_matches_replay"]
        or source_counts != copy_counts
    ):
        failures.append("integrity or projection-copy replay failed")
    return {
        "signature": signature,
        "semantic_digest": canonical_digest(signature),
        "failures": failures,
        "ok": not failures,
    }


def run_lab(corpus: Path = CORPUS) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="tsq-artifact-runner-lab-"
    ) as directory:
        root = Path(directory)
        first_root, second_root = root / "first", root / "second"
        first_root.mkdir()
        second_root.mkdir()
        first = run_once(first_root / "lab.db", first_root, corpus)
        second = run_once(second_root / "lab.db", second_root, corpus)
    stable = (
        first["semantic_digest"] == second["semantic_digest"]
        and first["signature"] == second["signature"]
    )
    failures = [*first["failures"], *second["failures"]]
    if not stable:
        failures.append("fresh runs produced different semantic digests")
    return {
        "lab_version": LAB_VERSION,
        "ok": first["ok"] and second["ok"] and stable,
        "semantic_digest_stable": stable,
        "first_semantic_digest": first["semantic_digest"],
        "second_semantic_digest": second["semantic_digest"],
        "stable_signature": first["signature"],
        "authority_boundary": first["signature"]["authority_boundary"],
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args(argv)
    report = run_lab(args.corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.stdout:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "status": "ok" if report["ok"] else "failed",
                    "semantic_digest": report["first_semantic_digest"],
                    "semantic_digest_stable": report[
                        "semantic_digest_stable"
                    ],
                    "shadow_only": True,
                    "mastery_applied": False,
                    "certification_applied": False,
                    "process_is_not_os_filesystem_network_sandbox": True,
                    "output": str(args.output),
                },
                sort_keys=True,
            )
        )
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
