#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0

"""Falsify scoring-claim recovery on disposable, event-backed databases.

The laboratory simulates two provider callbacks that fail after admission.  It
observes one as unknown before recovering a content-free scoring result after
the learner session has ended, and closes the other only through a synthetic
adapter explicitly capable of proving non-acceptance.  It never retries a
provider operation, grants scorer authority, or changes learner mastery.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tsq.corpus import read_and_parse  # noqa: E402
from tsq.engine import AdaptiveEngine  # noqa: E402
from tsq.errors import ConflictError, ValidationError  # noqa: E402
from tsq.evidence import (  # noqa: E402
    ActionPhase,
    EvaluationStatus,
    canonical_digest,
)
from tsq.performance import (  # noqa: E402
    ImportedCriterionResult,
    ImportedEvaluation,
    ScoringProviderRegistry,
    SyntheticDeterministicProvider,
)
from tsq.performance_ledger import (  # noqa: E402
    PerformanceLedger,
    read_task_release,
)
from tsq.reconciliation import (  # noqa: E402
    ReconciliationObservation,
    ReconciliationOutcome,
    ScoringReconciliationReceipt,
    ScoringReconciliationRegistry,
    SyntheticReconciliationAdapter,
)
from tsq.replay import ProjectionReplay  # noqa: E402
from tsq.store import Database  # noqa: E402


LAB_VERSION = "scoring-reconciliation-lab-v1"
START = datetime(2117, 5, 6, 9, 0, tzinfo=timezone.utc)
DEFAULT_CORPUS = PROJECT_ROOT / "corpus"
TASK_RELEASE = (
    PROJECT_ROOT / "tests" / "fixtures" / "reviewed_productive_task_release.json"
)
TASK_RELEASE_CORPUS_PLACEHOLDER = "rel_fixture_requires_explicit_pinning"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "scoring_reconciliation_lab.json"
)
SUBMISSION_DIGEST = "7" * 64


class _StrandedSyntheticProvider(SyntheticDeterministicProvider):
    """Fixture whose admitted callback never returns to the ledger."""

    def __init__(
        self,
        imported: ImportedEvaluation,
        *,
        provider_id: str,
    ) -> None:
        super().__init__(imported, provider_id=provider_id)
        self.calls = 0

    def score(self, request):
        self.calls += 1
        raise RuntimeError("synthetic response channel interrupted")


def _projection_boundary(
    database: Database,
    learner_id: str,
    session_id: str,
) -> dict[str, Any]:
    with database.read() as connection:
        return {
            "learner_revision": connection.execute(
                "SELECT revision FROM learners WHERE id=?",
                (learner_id,),
            ).fetchone()["revision"],
            "session_revision": connection.execute(
                "SELECT revision FROM sessions WHERE id=?",
                (session_id,),
            ).fetchone()["revision"],
            "learner_projection_hash": database.learner_projection_hash(
                learner_id,
                connection,
            ),
        }


def _imported_evaluation(
    task,
    submission_id: str,
    *,
    outcome_code: str,
    score: float,
) -> ImportedEvaluation:
    return ImportedEvaluation(
        criteria=tuple(
            ImportedCriterionResult(
                criterion_id=criterion.id,
                status=EvaluationStatus.VALID,
                score=score,
                outcome_code=outcome_code,
                phase=ActionPhase.UNASSISTED,
                source_action_ids=(submission_id,),
                reliability=0.9,
            )
            for criterion in task.criteria
        )
    )


def _observation(
    claim: dict[str, Any],
    *,
    outcome: ReconciliationOutcome,
    reconciler_id: str,
    observed_at: datetime,
    imported: ImportedEvaluation | None = None,
    completed_at: datetime | None = None,
) -> ReconciliationObservation:
    result_digest = None if imported is None else imported.digest
    provider_receipt_digest = canonical_digest(
        {
            "type": "tsq.scoring_reconciliation_lab_receipt",
            "claim_id": claim["id"],
            "provider_operation_digest": claim[
                "provider_operation_digest"
            ],
            "outcome": outcome.value,
            "observed_at": observed_at.isoformat(),
            "completed_at": (
                None if completed_at is None else completed_at.isoformat()
            ),
            "result_digest": result_digest,
        }
    )
    receipt = ScoringReconciliationReceipt(
        claim_id=claim["id"],
        attempt_id=claim["attempt_id"],
        evaluation_id=claim["evaluation_id"],
        through_sequence=claim["through_sequence"],
        provider_id=claim["provider_id"],
        provider_version=claim["provider_version"],
        reconciler_id=reconciler_id,
        reconciler_version="lab-v1",
        action_trace_digest=claim["action_trace_digest"],
        command_hash=claim["command_hash"],
        scoring_request_digest=claim["scoring_request_digest"],
        provider_binding_digest=claim["provider_binding_digest"],
        outcome=outcome,
        observed_at=observed_at.isoformat(),
        completed_at=(
            None if completed_at is None else completed_at.isoformat()
        ),
        result_digest=result_digest,
        reason_code={
            ReconciliationOutcome.UNKNOWN: "lab_lookup_ambiguous",
            ReconciliationOutcome.DEFINITELY_ABSENT: (
                "lab_operation_never_accepted"
            ),
            ReconciliationOutcome.COMPLETED: "lab_result_recovered",
        }[outcome],
        provider_operation_digest=claim["provider_operation_digest"],
        provider_receipt_digest=provider_receipt_digest,
        attestation_digest=canonical_digest(
            {
                "type": "tsq.scoring_reconciliation_lab_attestation",
                "reconciler_id": reconciler_id,
                "provider_receipt_digest": provider_receipt_digest,
                "synthetic": True,
            }
        ),
    )
    return ReconciliationObservation(
        receipt=receipt,
        imported_evaluation=imported,
    )


def _registry(
    observation: ReconciliationObservation,
    *,
    reconciler_id: str,
    can_prove_absence: bool,
) -> tuple[
    ScoringReconciliationRegistry,
    SyntheticReconciliationAdapter,
]:
    adapter = SyntheticReconciliationAdapter(
        observation,
        reconciler_id=reconciler_id,
        reconciler_version="lab-v1",
        can_prove_absence=can_prove_absence,
    )
    registry = ScoringReconciliationRegistry(allow_synthetic=True)
    registry.register(adapter, adapter.authority_binding)
    return registry, adapter


def run_once(
    database_path: Path,
    corpus_path: Path = DEFAULT_CORPUS,
) -> dict[str, Any]:
    database = Database(database_path)
    database.initialize()
    corpus_report = database.import_corpus(
        *read_and_parse(corpus_path, include_catalog=True)
    )
    engine = AdaptiveEngine(database)
    ledger = PerformanceLedger(database)
    template = read_task_release(TASK_RELEASE)
    if (
        template.corpus_release_id
        != TASK_RELEASE_CORPUS_PLACEHOLDER
    ):
        raise RuntimeError(
            "Productive-task fixture lost its explicit corpus placeholder."
        )
    release = replace(
        template,
        corpus_release_id=corpus_report["release_id"],
    )
    release_report = ledger.publish_release(release, now=START)
    task = release.tasks[0][1]

    def stranded_claim(
        *,
        learner_id: str,
        seed: int,
        prefix: str,
        provider_id: str,
        offset: int,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        _StrandedSyntheticProvider,
        ImportedEvaluation,
    ]:
        origin = START + timedelta(hours=offset)
        engine.create_learner(learner_id, learner_id)
        session = engine.start_session(
            learner_id,
            "t_transformers",
            seed=seed,
            now=origin,
        )
        attempt = ledger.start_attempt(
            session["id"],
            task.id,
            task_version=task.version,
            task_release_id=release_report["release_id"],
            idempotency_key=f"{prefix}-attempt",
            now=origin + timedelta(minutes=1),
        )
        submission = ledger.record_action(
            attempt["id"],
            "submitted",
            {"submission_digest": SUBMISSION_DIGEST},
            idempotency_key=f"{prefix}-submission",
            now=origin + timedelta(minutes=2),
        )
        imported = _imported_evaluation(
            task,
            submission["id"],
            outcome_code=f"{prefix}_recovered_result",
            score=0.8,
        )
        provider = _StrandedSyntheticProvider(
            imported,
            provider_id=provider_id,
        )
        registry = ScoringProviderRegistry(allow_synthetic=True)
        registry.register(provider, provider.authority_binding)
        try:
            ledger.score_attempt(
                attempt["id"],
                registry,
                provider.provider_id,
                provider.provider_version,
                idempotency_key=f"{prefix}-score",
                now=origin + timedelta(minutes=3),
            )
        except ValidationError:
            pass
        else:
            raise AssertionError(
                "Synthetic stranded callback unexpectedly returned."
            )
        claims = ledger.list_scoring_claims(
            attempt_id=attempt["id"],
            status="unreconciled",
        )
        if len(claims) != 1:
            raise AssertionError("Stranded callback did not leave one claim.")
        return session, claims[0], provider, imported

    completed_session, completed_claim, completed_provider, recovered = (
        stranded_claim(
            learner_id="reconciliation-lab-completed",
            seed=1901,
            prefix="reconciliation-lab-completed",
            provider_id="synthetic.reconciliation-lab-completed",
            offset=1,
        )
    )
    completed_origin = START + timedelta(hours=1)
    completed_before_unknown = _projection_boundary(
        database,
        "reconciliation-lab-completed",
        completed_session["id"],
    )
    unknown_registry, unknown_adapter = _registry(
        _observation(
            completed_claim,
            outcome=ReconciliationOutcome.UNKNOWN,
            reconciler_id="synthetic.reconciliation-lab-status",
            observed_at=completed_origin + timedelta(minutes=4),
        ),
        reconciler_id="synthetic.reconciliation-lab-status",
        can_prove_absence=False,
    )
    unknown = ledger.reconcile_scoring_claim(
        completed_claim["id"],
        unknown_registry,
        "synthetic.reconciliation-lab-status",
        "lab-v1",
        idempotency_key="reconciliation-lab-unknown",
        now=completed_origin + timedelta(minutes=4),
    )
    second_unknown_registry, second_unknown_adapter = _registry(
        _observation(
            completed_claim,
            outcome=ReconciliationOutcome.UNKNOWN,
            reconciler_id="synthetic.reconciliation-lab-status",
            observed_at=completed_origin
            + timedelta(minutes=4, seconds=15),
        ),
        reconciler_id="synthetic.reconciliation-lab-status",
        can_prove_absence=False,
    )
    second_unknown = ledger.reconcile_scoring_claim(
        completed_claim["id"],
        second_unknown_registry,
        "synthetic.reconciliation-lab-status",
        "lab-v1",
        idempotency_key="reconciliation-lab-unknown-again",
        now=completed_origin + timedelta(minutes=4, seconds=15),
    )
    completed_after_unknown = _projection_boundary(
        database,
        "reconciliation-lab-completed",
        completed_session["id"],
    )
    retry_registry = ScoringProviderRegistry(allow_synthetic=True)
    retry_registry.register(
        completed_provider,
        completed_provider.authority_binding,
    )
    try:
        ledger.score_attempt(
            completed_claim["attempt_id"],
            retry_registry,
            completed_provider.provider_id,
            completed_provider.provider_version,
            idempotency_key="reconciliation-lab-completed-score",
            now=completed_origin + timedelta(minutes=4, seconds=30),
        )
    except ConflictError:
        pass
    else:
        raise AssertionError("Unknown claim caused a scorer retry.")
    database.end_session(
        completed_session["id"],
        now=completed_origin + timedelta(minutes=5),
    )
    before_recovery = _projection_boundary(
        database,
        "reconciliation-lab-completed",
        completed_session["id"],
    )
    completed_registry, completed_adapter = _registry(
        _observation(
            completed_claim,
            outcome=ReconciliationOutcome.COMPLETED,
            reconciler_id="synthetic.reconciliation-lab-status",
            observed_at=completed_origin + timedelta(minutes=6),
            imported=recovered,
            completed_at=completed_origin + timedelta(
                minutes=3,
                seconds=30,
            ),
        ),
        reconciler_id="synthetic.reconciliation-lab-status",
        can_prove_absence=False,
    )
    completed = ledger.reconcile_scoring_claim(
        completed_claim["id"],
        completed_registry,
        "synthetic.reconciliation-lab-status",
        "lab-v1",
        idempotency_key="reconciliation-lab-completed-observation",
        now=completed_origin + timedelta(minutes=6),
    )
    after_recovery = _projection_boundary(
        database,
        "reconciliation-lab-completed",
        completed_session["id"],
    )

    absent_session, absent_claim, absent_provider, _absent_result = (
        stranded_claim(
            learner_id="reconciliation-lab-absent",
            seed=1902,
            prefix="reconciliation-lab-absent",
            provider_id="synthetic.reconciliation-lab-absent",
            offset=2,
        )
    )
    absent_origin = START + timedelta(hours=2)
    database.end_session(
        absent_session["id"],
        now=absent_origin + timedelta(minutes=4),
    )
    before_absence = _projection_boundary(
        database,
        "reconciliation-lab-absent",
        absent_session["id"],
    )
    absent_registry, absent_adapter = _registry(
        _observation(
            absent_claim,
            outcome=ReconciliationOutcome.DEFINITELY_ABSENT,
            reconciler_id="synthetic.reconciliation-lab-fence",
            observed_at=absent_origin + timedelta(minutes=5),
        ),
        reconciler_id="synthetic.reconciliation-lab-fence",
        can_prove_absence=True,
    )
    absent = ledger.reconcile_scoring_claim(
        absent_claim["id"],
        absent_registry,
        "synthetic.reconciliation-lab-fence",
        "lab-v1",
        idempotency_key="reconciliation-lab-absence",
        now=absent_origin + timedelta(minutes=5),
    )
    after_absence = _projection_boundary(
        database,
        "reconciliation-lab-absent",
        absent_session["id"],
    )
    absent_retry_registry = ScoringProviderRegistry(allow_synthetic=True)
    absent_retry_registry.register(
        absent_provider,
        absent_provider.authority_binding,
    )
    try:
        ledger.score_attempt(
            absent_claim["attempt_id"],
            absent_retry_registry,
            absent_provider.provider_id,
            absent_provider.provider_version,
            idempotency_key="reconciliation-lab-absent-score",
            now=absent_origin + timedelta(minutes=6),
        )
    except ConflictError:
        pass
    else:
        raise AssertionError("Definitely-absent claim caused a scorer retry.")

    completed_report = ledger.report(completed_claim["attempt_id"])
    absent_report = ledger.report(absent_claim["attempt_id"])
    completed_replay = ProjectionReplay(database).check(
        "reconciliation-lab-completed"
    )
    absent_replay = ProjectionReplay(database).check(
        "reconciliation-lab-absent"
    )
    integrity = database.verify_integrity()
    with database.read() as connection:
        event_counts = {
            row["event_type"]: row["n"]
            for row in connection.execute(
                """SELECT event_type, COUNT(*) AS n
                   FROM events
                   WHERE event_type IN (
                       'PerformanceScoringClaimed',
                       'PerformanceScoringReconciled',
                       'TaskEvaluationRecorded',
                       'ShadowEvidenceReduced'
                   )
                   GROUP BY event_type ORDER BY event_type"""
            )
        }
        recovery_events = connection.execute(
            """SELECT event_type, session_id
               FROM events
               WHERE correlation_id=?
                 AND event_type IN (
                     'PerformanceScoringReconciled',
                     'TaskEvaluationRecorded',
                     'ShadowEvidenceReduced'
                 )
               ORDER BY stream_version""",
            (completed_claim["attempt_id"],),
        ).fetchall()
        reconciliation_metadata = [
            json.loads(row["metadata_json"])
            for row in connection.execute(
                """SELECT metadata_json FROM events
                   WHERE event_type='PerformanceScoringReconciled'
                   ORDER BY stream_id, stream_version"""
            )
        ]

    stable = {
        "unknown_status": unknown["status"],
        "unknown_terminal": unknown["terminal"],
        "repeated_unknown_count": second_unknown[
            "reconciliation_count"
        ],
        "completed_status": completed["status"],
        "completed_terminal": completed["terminal"],
        "absent_status": absent["status"],
        "absent_terminal": absent["terminal"],
        "completed_provider_calls": completed_provider.calls,
        "absent_provider_calls": absent_provider.calls,
        "unknown_lookup_calls": (
            unknown_adapter.lookup_calls
            + second_unknown_adapter.lookup_calls
        ),
        "completed_lookup_calls": completed_adapter.lookup_calls,
        "absent_lookup_calls": absent_adapter.lookup_calls,
        "unknown_projection_unchanged": (
            completed_before_unknown == completed_after_unknown
        ),
        "recovery_projection_unchanged": before_recovery == after_recovery,
        "absence_projection_unchanged": before_absence == after_absence,
        "completed_evaluation_count": completed_report[
            "evaluation_count"
        ],
        "absent_evaluation_count": absent_report["evaluation_count"],
        "recovered_shadow_weight": completed[
            "shadow_evidence"
        ]["total_evidence_weight"],
        "recovery_events_session_null": all(
            row["session_id"] is None for row in recovery_events
        ),
        "recovery_event_types": [
            row["event_type"] for row in recovery_events
        ],
        "automatic_retry_disabled": all(
            metadata["automatic_retry_allowed"] is False
            for metadata in reconciliation_metadata
        ),
        "projection_disabled": all(
            metadata["projection_applied"] is False
            for metadata in reconciliation_metadata
        ),
        "certification_disabled": all(
            metadata["certification_applied"] is False
            for metadata in reconciliation_metadata
        ),
        "skill_authority_disabled": all(
            metadata["skill_authority"] is False
            for metadata in reconciliation_metadata
        ),
        "observational_only": all(
            metadata["observational_only"] is True
            for metadata in reconciliation_metadata
        ),
        "event_counts": event_counts,
        "completed_replay_ok": completed_replay["ok"],
        "absent_replay_ok": absent_replay["ok"],
        "performance_replay_ok": (
            completed_replay["performance_projection_matches_replay"]
            and absent_replay["performance_projection_matches_replay"]
        ),
        "integrity_ok": integrity["ok"],
    }
    failures: list[str] = []
    if stable["unknown_status"] != "unknown" or stable["unknown_terminal"]:
        failures.append("Unknown observation closed its scoring claim.")
    if stable["repeated_unknown_count"] != 2:
        failures.append("Repeated uncertainty was not preserved.")
    if (
        stable["completed_status"] != "completed"
        or not stable["completed_terminal"]
    ):
        failures.append("Recovered result did not close its scoring claim.")
    if (
        stable["absent_status"] != "definitely_absent"
        or not stable["absent_terminal"]
    ):
        failures.append("Absence fence did not close its scoring claim.")
    if (
        stable["completed_provider_calls"] != 1
        or stable["absent_provider_calls"] != 1
    ):
        failures.append("A provider callback was retried.")
    if (
        stable["unknown_lookup_calls"] != 2
        or stable["completed_lookup_calls"] != 1
        or stable["absent_lookup_calls"] != 1
    ):
        failures.append("An observational lookup was skipped or repeated.")
    if not all(
        (
            stable["unknown_projection_unchanged"],
            stable["recovery_projection_unchanged"],
            stable["absence_projection_unchanged"],
        )
    ):
        failures.append("Reconciliation changed a learner projection.")
    if stable["recovered_shadow_weight"] != 0.0:
        failures.append("Recovered result acquired evidence authority.")
    if stable["completed_evaluation_count"] != 1:
        failures.append("Recovered result did not produce exactly one evaluation.")
    if stable["absent_evaluation_count"] != 0:
        failures.append("Definitely-absent operation acquired an evaluation.")
    if stable["event_counts"] != {
        "PerformanceScoringClaimed": 2,
        "PerformanceScoringReconciled": 4,
        "ShadowEvidenceReduced": 1,
        "TaskEvaluationRecorded": 1,
    }:
        failures.append("Scoring reconciliation event cardinality drifted.")
    if stable["recovery_event_types"] != [
        "PerformanceScoringReconciled",
        "PerformanceScoringReconciled",
        "PerformanceScoringReconciled",
        "TaskEvaluationRecorded",
        "ShadowEvidenceReduced",
    ]:
        failures.append("Recovered event sequence is not append-only and exact.")
    if (
        not stable["recovery_events_session_null"]
        or not stable["automatic_retry_disabled"]
        or not stable["projection_disabled"]
        or not stable["certification_disabled"]
        or not stable["skill_authority_disabled"]
        or not stable["observational_only"]
    ):
        failures.append("Recovered events crossed a fail-closed boundary.")
    if (
        not stable["completed_replay_ok"]
        or not stable["absent_replay_ok"]
        or not stable["performance_replay_ok"]
        or not stable["integrity_ok"]
    ):
        failures.append("Integrity or projection-copy replay failed.")
    return {
        "lab_version": LAB_VERSION,
        "database": str(database_path),
        "stable_signature": stable,
        "stable_digest": canonical_digest(stable),
        "failures": failures,
        "ok": not failures,
    }


def run_lab(corpus_path: Path = DEFAULT_CORPUS) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="tsq-scoring-reconciliation-lab-"
    ) as directory:
        root = Path(directory)
        first = run_once(root / "first.db", corpus_path)
        second = run_once(root / "second.db", corpus_path)
    deterministic = (
        first["stable_signature"] == second["stable_signature"]
    )
    failures = [*first["failures"], *second["failures"]]
    if not deterministic:
        failures.append("Independent disposable reruns diverged.")
    return {
        "lab_version": LAB_VERSION,
        "deterministic_rerun": deterministic,
        "stable_digest": first["stable_digest"],
        "stable_signature": first["stable_signature"],
        "failures": failures,
        "ok": first["ok"] and second["ok"] and deterministic,
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
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.stdout:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "status": "ok" if result["ok"] else "failed",
                    "deterministic_rerun": result[
                        "deterministic_rerun"
                    ],
                    "stable_digest": result["stable_digest"],
                    "output": str(args.output),
                },
                sort_keys=True,
            )
        )
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
