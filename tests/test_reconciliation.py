# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import json
import unittest
from dataclasses import replace

from tsq.evidence import (
    ActionPhase,
    EvaluationStatus,
    canonical_digest,
    canonical_json,
)
from tsq.performance import ImportedCriterionResult, ImportedEvaluation
from tsq.reconciliation import (
    ReconcilerExecutionError,
    ReconcilerNotFoundError,
    ReconciliationAuthorityBinding,
    ReconciliationObservation,
    ReconciliationOutcome,
    ReconciliationProtocolError,
    ReconciliationResult,
    RegisteredReconciler,
    ScoringReconciliationReceipt,
    ScoringReconciliationRegistry,
    ScoringReconciliationRequest,
    SyntheticReconciliationAdapter,
    TaskScoringReconciler,
    provider_scoring_operation_digest,
)


_D0 = "0" * 64
_D1 = "1" * 64
_D2 = "2" * 64
_D3 = "3" * 64
_D4 = "4" * 64
_D5 = "5" * 64
_D6 = "6" * 64
_D7 = "7" * 64
_D8 = "8" * 64
_D9 = "9" * 64

_OBSERVED_AT = "2026-07-27T12:00:00+00:00"
_COMPLETED_AT = "2026-07-27T11:59:00+00:00"


def imported_evaluation() -> ImportedEvaluation:
    return ImportedEvaluation(
        criteria=(
            ImportedCriterionResult(
                criterion_id="criterion_operation",
                status=EvaluationStatus.VALID,
                score=0.75,
                outcome_code="fixture_observed",
                phase=ActionPhase.UNASSISTED,
                source_action_ids=("action_submitted",),
                reliability=0.9,
            ),
        )
    )


def reconciliation_request(**overrides) -> ScoringReconciliationRequest:
    values = {
        "claim_id": "claim_scoring_1",
        "attempt_id": "attempt_productive_1",
        "evaluation_id": "evaluation_productive_1",
        "through_sequence": 4,
        "provider_id": "provider.fixture",
        "provider_version": "v1",
        "action_trace_digest": _D0,
        "command_hash": _D1,
        "scoring_request_digest": _D2,
        "provider_binding_digest": _D3,
    }
    values.update(overrides)
    values.setdefault(
        "provider_operation_digest",
        provider_scoring_operation_digest(
            claim_id=values["claim_id"],
            evaluation_id=values["evaluation_id"],
            scoring_request_digest=values["scoring_request_digest"],
            provider_binding_digest=values["provider_binding_digest"],
        ),
    )
    return ScoringReconciliationRequest(**values)


def reconciliation_receipt(
    outcome: ReconciliationOutcome = ReconciliationOutcome.UNKNOWN,
    *,
    request: ScoringReconciliationRequest | None = None,
    reconciler_id: str = "reconciler.fixture",
    reconciler_version: str = "v1",
    result: ImportedEvaluation | None = None,
    **overrides,
) -> ScoringReconciliationReceipt:
    boundary = request or reconciliation_request()
    values = {
        "claim_id": boundary.claim_id,
        "attempt_id": boundary.attempt_id,
        "evaluation_id": boundary.evaluation_id,
        "through_sequence": boundary.through_sequence,
        "provider_id": boundary.provider_id,
        "provider_version": boundary.provider_version,
        "reconciler_id": reconciler_id,
        "reconciler_version": reconciler_version,
        "action_trace_digest": boundary.action_trace_digest,
        "command_hash": boundary.command_hash,
        "scoring_request_digest": boundary.scoring_request_digest,
        "provider_binding_digest": boundary.provider_binding_digest,
        "outcome": outcome,
        "observed_at": _OBSERVED_AT,
        "completed_at": (
            _COMPLETED_AT
            if outcome is ReconciliationOutcome.COMPLETED
            else None
        ),
        "result_digest": (
            result.digest
            if outcome is ReconciliationOutcome.COMPLETED and result is not None
            else None
        ),
        "reason_code": {
            ReconciliationOutcome.UNKNOWN: "provider_lookup_ambiguous",
            ReconciliationOutcome.DEFINITELY_ABSENT: (
                "provider_operation_never_accepted"
            ),
            ReconciliationOutcome.COMPLETED: "provider_result_recovered",
        }[outcome],
        "provider_receipt_digest": _D4,
        "attestation_digest": _D5,
    }
    explicit_operation_digest = overrides.pop(
        "provider_operation_digest", None
    )
    values.update(overrides)
    values["provider_operation_digest"] = (
        explicit_operation_digest
        if explicit_operation_digest is not None
        else provider_scoring_operation_digest(
            claim_id=values["claim_id"],
            evaluation_id=values["evaluation_id"],
            scoring_request_digest=values["scoring_request_digest"],
            provider_binding_digest=values["provider_binding_digest"],
        )
    )
    return ScoringReconciliationReceipt(**values)


def observation(
    outcome: ReconciliationOutcome = ReconciliationOutcome.UNKNOWN,
    *,
    request: ScoringReconciliationRequest | None = None,
    reconciler_id: str = "reconciler.fixture",
    reconciler_version: str = "v1",
) -> ReconciliationObservation:
    result = (
        imported_evaluation()
        if outcome is ReconciliationOutcome.COMPLETED
        else None
    )
    return ReconciliationObservation(
        receipt=reconciliation_receipt(
            outcome,
            request=request,
            reconciler_id=reconciler_id,
            reconciler_version=reconciler_version,
            result=result,
        ),
        imported_evaluation=result,
    )


class FixedObservationalAdapter:
    synthetic = False

    def __init__(
        self,
        fixed: ReconciliationObservation | object,
        *,
        provider_id: str = "provider.fixture",
        provider_version: str = "v1",
        reconciler_id: str = "reconciler.fixture",
        reconciler_version: str = "v1",
        can_prove_absence: bool = False,
    ) -> None:
        self.provider_id = provider_id
        self.provider_version = provider_version
        self.reconciler_id = reconciler_id
        self.reconciler_version = reconciler_version
        self.can_prove_absence = can_prove_absence
        self.fixed = fixed
        self.failure: Exception | None = None
        self.lookup_calls = 0
        self.mutate_during_lookup = False

    def lookup(
        self, request: ScoringReconciliationRequest
    ) -> ReconciliationObservation:
        self.lookup_calls += 1
        if self.failure is not None:
            raise self.failure
        if self.mutate_during_lookup:
            self.reconciler_version = "mutated-v2"
        return self.fixed  # type: ignore[return-value]


def authority(
    adapter: FixedObservationalAdapter,
) -> ReconciliationAuthorityBinding:
    return ReconciliationAuthorityBinding(
        provider_id=adapter.provider_id,
        provider_version=adapter.provider_version,
        reconciler_id=adapter.reconciler_id,
        reconciler_version=adapter.reconciler_version,
        manifest_digest=_D8,
        synthetic=adapter.synthetic,
        can_prove_absence=adapter.can_prove_absence,
    )


class ReconciliationReceiptTests(unittest.TestCase):
    def test_closed_outcomes_have_exact_result_lifecycle(self) -> None:
        result = imported_evaluation()
        completed = reconciliation_receipt(
            ReconciliationOutcome.COMPLETED,
            result=result,
        )
        absent = reconciliation_receipt(
            ReconciliationOutcome.DEFINITELY_ABSENT
        )
        unknown = reconciliation_receipt(ReconciliationOutcome.UNKNOWN)

        self.assertEqual(completed.result_digest, result.digest)
        self.assertEqual(completed.completed_at, _COMPLETED_AT)
        for receipt in (absent, unknown):
            self.assertIsNone(receipt.result_digest)
            self.assertIsNone(receipt.completed_at)
        with self.assertRaisesRegex(
            ReconciliationProtocolError, "requires completed_at"
        ):
            reconciliation_receipt(
                ReconciliationOutcome.COMPLETED,
                completed_at=None,
                result_digest=None,
            )
        with self.assertRaisesRegex(
            ReconciliationProtocolError, "forbid completed_at"
        ):
            reconciliation_receipt(
                ReconciliationOutcome.UNKNOWN,
                completed_at=_COMPLETED_AT,
                result_digest=_D6,
            )

    def test_receipt_requires_canonical_aware_ordered_timestamps(self) -> None:
        for observed_at in (
            "2026-07-27T12:00:00",
            "2026-07-27T05:00:00-07:00",
            "2026-07-27T12:00:00Z",
        ):
            with self.subTest(observed_at=observed_at):
                with self.assertRaises(ReconciliationProtocolError):
                    reconciliation_receipt(observed_at=observed_at)
        with self.assertRaisesRegex(
            ReconciliationProtocolError, "before it occurred"
        ):
            reconciliation_receipt(
                ReconciliationOutcome.COMPLETED,
                result=imported_evaluation(),
                completed_at="2026-07-27T12:01:00+00:00",
            )

    def test_receipt_requires_lowercase_sha256_and_stable_reason_id(self) -> None:
        for field, value in (
            ("action_trace_digest", "A" * 64),
            ("command_hash", "not-a-digest"),
            ("scoring_request_digest", "2" * 63),
            ("provider_binding_digest", "G" * 64),
            ("provider_receipt_digest", ""),
            ("attestation_digest", "5" * 65),
            ("reason_code", "free form reason"),
        ):
            with self.subTest(field=field):
                with self.assertRaises(ReconciliationProtocolError):
                    reconciliation_receipt(**{field: value})

    def test_operation_digest_has_one_exported_canonical_formula(self) -> None:
        request = reconciliation_request()
        expected = canonical_digest(
            {
                "type": "tsq.provider_scoring_operation",
                "claim_id": request.claim_id,
                "evaluation_id": request.evaluation_id,
                "scoring_request_digest": request.scoring_request_digest,
                "provider_binding_digest": request.provider_binding_digest,
            }
        )
        self.assertEqual(request.provider_operation_digest, expected)
        with self.assertRaisesRegex(
            ReconciliationProtocolError, "does not match"
        ):
            reconciliation_request(provider_operation_digest=_D9)
        with self.assertRaisesRegex(
            ReconciliationProtocolError, "does not match"
        ):
            reconciliation_receipt(provider_operation_digest=_D9)

    def test_receipt_strict_json_round_trip_rejects_schema_drift(self) -> None:
        receipt = reconciliation_receipt()
        encoded = canonical_json(receipt.terms())
        decoded = ScoringReconciliationReceipt.from_json(encoded)
        self.assertEqual(decoded, receipt)
        self.assertEqual(decoded.digest, receipt.digest)

        duplicate = '{"claim_id":"duplicate",' + encoded[1:]
        with self.assertRaisesRegex(
            ReconciliationProtocolError, "duplicate field"
        ):
            ScoringReconciliationReceipt.from_json(duplicate)
        extra = receipt.terms()
        extra["future_field"] = True
        with self.assertRaisesRegex(
            ReconciliationProtocolError, "unexpected future_field"
        ):
            ScoringReconciliationReceipt.from_terms(extra)
        unknown = receipt.terms()
        unknown["outcome"] = "maybe"
        with self.assertRaisesRegex(
            ReconciliationProtocolError, "unknown value"
        ):
            ScoringReconciliationReceipt.from_terms(unknown)

    def test_observation_requires_result_exactly_for_completion(self) -> None:
        result = imported_evaluation()
        completed = reconciliation_receipt(
            ReconciliationOutcome.COMPLETED,
            result=result,
        )
        with self.assertRaisesRegex(
            ReconciliationProtocolError, "requires an ImportedEvaluation"
        ):
            ReconciliationObservation(completed)
        mismatched = reconciliation_receipt(
            ReconciliationOutcome.COMPLETED,
            result=result,
            result_digest=_D9,
        )
        with self.assertRaisesRegex(
            ReconciliationProtocolError, "digest does not match"
        ):
            ReconciliationObservation(mismatched, result)
        with self.assertRaisesRegex(
            ReconciliationProtocolError, "forbid a result"
        ):
            ReconciliationObservation(
                reconciliation_receipt(ReconciliationOutcome.UNKNOWN),
                result,
            )


class ReconciliationAuthorityTests(unittest.TestCase):
    def test_authority_and_registered_identity_are_canonical_and_separate(self) -> None:
        adapter = FixedObservationalAdapter(observation())
        binding = authority(adapter)
        decoded = ReconciliationAuthorityBinding.from_terms(binding.terms())
        self.assertEqual(decoded, binding)
        registered = RegisteredReconciler(
            provider_id=binding.provider_id,
            provider_version=binding.provider_version,
            reconciler_id=binding.reconciler_id,
            reconciler_version=binding.reconciler_version,
            manifest_digest=binding.manifest_digest,
            binding_digest=binding.digest,
            synthetic=False,
            can_prove_absence=False,
        )
        self.assertEqual(
            RegisteredReconciler.from_terms(registered.terms()),
            registered,
        )
        self.assertTrue(registered.terms()["observational_only"])
        self.assertFalse(registered.terms()["skill_authority"])
        with self.assertRaisesRegex(
            ReconciliationProtocolError, "binding digest"
        ):
            replace(registered, binding_digest=_D9)

    def test_synthetic_registration_is_explicit_and_namespace_bound(self) -> None:
        fixed = observation(
            reconciler_id="synthetic.fixed-reconciler",
            reconciler_version="test-v1",
        )
        adapter = SyntheticReconciliationAdapter(fixed)
        with self.assertRaisesRegex(
            ReconciliationProtocolError, "allow_synthetic"
        ):
            ScoringReconciliationRegistry().register(
                adapter, adapter.authority_binding
            )
        registry = ScoringReconciliationRegistry(allow_synthetic=True)
        registered = registry.register(adapter, adapter.authority_binding)
        self.assertTrue(registered.synthetic)
        with self.assertRaisesRegex(
            ReconciliationProtocolError, "must start"
        ):
            ReconciliationAuthorityBinding(
                provider_id="provider.fixture",
                provider_version="v1",
                reconciler_id="reconciler.not-synthetic-namespace",
                reconciler_version="v1",
                manifest_digest=_D8,
                synthetic=True,
            )

    def test_synthetic_manifest_is_stable_across_observations(self) -> None:
        unknown = SyntheticReconciliationAdapter(
            observation(
                ReconciliationOutcome.UNKNOWN,
                reconciler_id="synthetic.stable-reconciler",
            ),
            reconciler_id="synthetic.stable-reconciler",
        )
        completed = SyntheticReconciliationAdapter(
            observation(
                ReconciliationOutcome.COMPLETED,
                reconciler_id="synthetic.stable-reconciler",
            ),
            reconciler_id="synthetic.stable-reconciler",
        )
        self.assertNotEqual(unknown._observation.digest, completed._observation.digest)
        self.assertEqual(
            unknown.authority_binding,
            completed.authority_binding,
        )


class ReconciliationRegistryTests(unittest.TestCase):
    def registered(
        self,
        fixed: ReconciliationObservation,
        *,
        can_prove_absence: bool = False,
    ) -> tuple[
        ScoringReconciliationRegistry,
        FixedObservationalAdapter,
        RegisteredReconciler,
    ]:
        adapter = FixedObservationalAdapter(
            fixed, can_prove_absence=can_prove_absence
        )
        registry = ScoringReconciliationRegistry()
        summary = registry.register(adapter, authority(adapter))
        return registry, adapter, summary

    def test_lookup_reconcile_is_observational_and_never_exposes_score(self) -> None:
        request = reconciliation_request()
        registry, adapter, _ = self.registered(observation(request=request))
        self.assertIsInstance(adapter, TaskScoringReconciler)
        self.assertFalse(hasattr(adapter, "score"))

        result = registry.reconcile(
            adapter.reconciler_id,
            adapter.reconciler_version,
            request,
        )
        self.assertEqual(result.outcome, ReconciliationOutcome.UNKNOWN)
        self.assertEqual(adapter.lookup_calls, 1)
        terms = result.terms()
        self.assertTrue(terms["observational_only"])
        self.assertFalse(terms["automatic_retry_allowed"])
        self.assertFalse(terms["projection_applied"])
        self.assertFalse(terms["certification_applied"])
        self.assertFalse(terms["skill_authority"])
        self.assertFalse(terms["cryptographic_verification_claim"])
        self.assertEqual(
            terms["attestation_semantics"],
            "registered_adapter_commitment",
        )
        self.assertEqual(ReconciliationResult.from_terms(terms), result)

    def test_completed_result_is_canonical_and_digest_bound(self) -> None:
        request = reconciliation_request()
        fixed = observation(
            ReconciliationOutcome.COMPLETED, request=request
        )
        registry, adapter, _ = self.registered(fixed)
        result = registry.reconcile(
            adapter.reconciler_id,
            adapter.reconciler_version,
            request,
        )
        self.assertEqual(result.outcome, ReconciliationOutcome.COMPLETED)
        self.assertEqual(
            result.imported_evaluation.digest,
            fixed.receipt.result_digest,
        )
        self.assertEqual(
            ReconciliationObservation.from_terms(fixed.terms()),
            fixed,
        )

    def test_exact_request_receipt_boundary_is_fail_closed(self) -> None:
        request = reconciliation_request()
        cases = (
            ("claim_id", "claim_other"),
            ("attempt_id", "attempt_other"),
            ("evaluation_id", "evaluation_other"),
            ("through_sequence", 5),
            ("action_trace_digest", _D6),
            ("command_hash", _D7),
            ("scoring_request_digest", _D6),
            ("provider_binding_digest", _D7),
        )
        for field, value in cases:
            with self.subTest(field=field):
                fixed = ReconciliationObservation(
                    reconciliation_receipt(
                        request=request, **{field: value}
                    )
                )
                registry, adapter, _ = self.registered(fixed)
                with self.assertRaisesRegex(
                    ReconciliationProtocolError, "exact claim request"
                ):
                    registry.reconcile(
                        adapter.reconciler_id,
                        adapter.reconciler_version,
                        request,
                    )

    def test_provider_and_reconciler_authority_mismatch_is_rejected(self) -> None:
        request = reconciliation_request()
        for field, value in (
            ("provider_id", "provider.other"),
            ("provider_version", "v2"),
            ("reconciler_id", "reconciler.other"),
            ("reconciler_version", "v2"),
        ):
            with self.subTest(field=field):
                fixed = ReconciliationObservation(
                    reconciliation_receipt(
                        request=request, **{field: value}
                    )
                )
                registry, adapter, _ = self.registered(fixed)
                expected = (
                    "exact claim request"
                    if field.startswith("provider_")
                    else "registered authority"
                )
                with self.assertRaisesRegex(
                    ReconciliationProtocolError, expected
                ):
                    registry.reconcile(
                        adapter.reconciler_id,
                        adapter.reconciler_version,
                        request,
                    )

    def test_definite_absence_requires_registered_guarantee(self) -> None:
        request = reconciliation_request()
        fixed = observation(
            ReconciliationOutcome.DEFINITELY_ABSENT,
            request=request,
        )
        registry, adapter, _ = self.registered(fixed)
        with self.assertRaisesRegex(
            ReconciliationProtocolError, "must remain unknown"
        ):
            registry.reconcile(
                adapter.reconciler_id,
                adapter.reconciler_version,
                request,
            )

        trusted, capable, _ = self.registered(
            fixed, can_prove_absence=True
        )
        result = trusted.reconcile(
            capable.reconciler_id,
            capable.reconciler_version,
            request,
        )
        self.assertEqual(
            result.outcome, ReconciliationOutcome.DEFINITELY_ABSENT
        )
        self.assertFalse(result.terms()["automatic_retry_allowed"])

    def test_direct_result_construction_cannot_bypass_registry_binding(self) -> None:
        request = reconciliation_request()
        fixed = observation(request=request)
        registry, _, summary = self.registered(fixed)
        self.assertEqual(len(registry.list()), 1)
        forged = ReconciliationObservation(
            reconciliation_receipt(
                request=request, claim_id="claim_forged"
            )
        )
        with self.assertRaisesRegex(
            ReconciliationProtocolError, "exact claim request"
        ):
            ReconciliationResult(request, forged, summary)

    def test_registration_is_exact_version_no_overwrite_and_sorted(self) -> None:
        fixed = observation()
        registry, adapter, summary = self.registered(fixed)
        with self.assertRaisesRegex(
            ReconciliationProtocolError, "already registered"
        ):
            registry.register(adapter, authority(adapter))
        second = FixedObservationalAdapter(
            fixed,
            provider_id="provider.alpha",
        )
        second_fixed = ReconciliationObservation(
            reconciliation_receipt(provider_id="provider.alpha")
        )
        second.fixed = second_fixed
        second_summary = registry.register(second, authority(second))
        self.assertEqual(
            [item.key for item in registry.list()],
            sorted((summary.key, second_summary.key)),
        )
        self.assertEqual(
            registry.inspect(*summary.key),
            summary,
        )
        with self.assertRaises(ReconcilerNotFoundError):
            registry.inspect(
                "provider.missing", "v1", "reconciler.fixture", "v1"
            )

    def test_adapter_identity_is_rechecked_before_and_after_lookup(self) -> None:
        request = reconciliation_request()
        fixed = observation(request=request)
        registry, adapter, _ = self.registered(fixed)
        adapter.provider_version = "mutated-v2"
        with self.assertRaisesRegex(
            ReconcilerExecutionError, "changed after registration"
        ):
            registry.reconcile("reconciler.fixture", "v1", request)
        self.assertEqual(adapter.lookup_calls, 0)

        registry, adapter, _ = self.registered(fixed)
        adapter.score = lambda _request: imported_evaluation()
        with self.assertRaisesRegex(
            ReconcilerExecutionError, "became unavailable"
        ):
            registry.reconcile("reconciler.fixture", "v1", request)
        self.assertEqual(adapter.lookup_calls, 0)

        registry, adapter, _ = self.registered(fixed)
        adapter.mutate_during_lookup = True
        with self.assertRaisesRegex(
            ReconcilerExecutionError, "changed during lookup"
        ):
            registry.reconcile("reconciler.fixture", "v1", request)
        self.assertEqual(adapter.lookup_calls, 1)

    def test_adapter_failures_and_nonprotocol_results_are_wrapped(self) -> None:
        request = reconciliation_request()
        fixed = observation(request=request)
        registry, adapter, _ = self.registered(fixed)
        adapter.failure = RuntimeError("external status store unavailable")
        with self.assertRaisesRegex(
            ReconcilerExecutionError, "lookup failed"
        ):
            registry.reconcile("reconciler.fixture", "v1", request)

        bad = FixedObservationalAdapter(object())
        registry = ScoringReconciliationRegistry()
        registry.register(bad, authority(bad))
        with self.assertRaisesRegex(
            ReconcilerExecutionError, "non-protocol observation"
        ):
            registry.reconcile("reconciler.fixture", "v1", request)

    def test_registration_rejects_score_capability_and_authority_drift(self) -> None:
        class ScoringAdapter(FixedObservationalAdapter):
            def score(self, request):
                raise AssertionError("must never be called")

        scorer = ScoringAdapter(observation())
        with self.assertRaisesRegex(
            ReconciliationProtocolError, "must not expose a score method"
        ):
            ScoringReconciliationRegistry().register(
                scorer, authority(scorer)
            )

        adapter = FixedObservationalAdapter(observation())
        mismatched = replace(authority(adapter), can_prove_absence=True)
        with self.assertRaisesRegex(
            ReconciliationProtocolError, "does not match"
        ):
            ScoringReconciliationRegistry().register(adapter, mismatched)


if __name__ == "__main__":
    unittest.main()
