# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import json
import unittest
from dataclasses import replace

from tsq.evidence import (
    ActionKind,
    ActionPhase,
    EvaluationStatus,
    ScorerContract,
    ScorerKind,
    TaskEvaluation,
)
from tsq.performance import (
    ImportedCriterionResult,
    ImportedEvaluation,
    CriterionAuthorityDecision,
    NormalizationMode,
    NormalizedScoringResult,
    ProviderAuthorityBinding,
    ProviderExecutionError,
    ProviderNotFoundError,
    RegisteredProvider,
    ScoringProtocolError,
    ScoringProviderRegistry,
    ScoringRequest,
    SyntheticDeterministicProvider,
    TaskScoringProvider,
    normalize_imported_evaluation,
)


_D0 = "0" * 64
_D1 = "1" * 64
_D2 = "2" * 64
_D3 = "3" * 64
_D4 = "4" * 64


def request(
    *criterion_ids: str,
    scorer_contract: ScorerContract | None = None,
) -> ScoringRequest:
    return ScoringRequest(
        evaluation_id="evaluation_perf_1",
        trace_id="trace_perf_1",
        task_id="task_perf_1",
        task_version=2,
        task_digest=_D0,
        action_trace_digest=_D1,
        criterion_ids=tuple(criterion_ids),
        scorer_contract=scorer_contract,
    )


def criterion(
    criterion_id: str,
    *,
    score: float | None = 0.75,
    status: EvaluationStatus = EvaluationStatus.VALID,
    attestation_digest: str | None = None,
) -> ImportedCriterionResult:
    return ImportedCriterionResult(
        criterion_id=criterion_id,
        status=status,
        score=score,
        outcome_code="observed",
        phase=ActionPhase.UNASSISTED,
        source_action_ids=("action_1",),
        attestation_digest=attestation_digest,
        misconception_ids=(),
        reliability=0.9,
    )


def imported(*criterion_ids: str, attested: bool = False) -> ImportedEvaluation:
    return ImportedEvaluation(
        criteria=tuple(
            criterion(
                criterion_id,
                attestation_digest=_D4 if attested else None,
            )
            for criterion_id in criterion_ids
        )
    )


class FixedProvider:
    synthetic = False

    def __init__(
        self,
        provider_id: str,
        provider_version: str,
        declared_kind: ScorerKind,
        result: ImportedEvaluation | object,
    ) -> None:
        self.provider_id = provider_id
        self.provider_version = provider_version
        self.declared_kind = declared_kind
        self.result = result
        self.failure: Exception | None = None

    def score(self, scoring_request: ScoringRequest) -> ImportedEvaluation:
        if self.failure is not None:
            raise self.failure
        return self.result  # type: ignore[return-value]


def binding(
    provider: FixedProvider,
    *,
    verified: bool,
    with_check_manifest: bool = False,
) -> ProviderAuthorityBinding:
    return ProviderAuthorityBinding(
        provider_id=provider.provider_id,
        provider_version=provider.provider_version,
        declared_kind=provider.declared_kind,
        authority_id=f"authority.{provider.provider_id}",
        authority_manifest_digest=_D2,
        check_set_manifests=(
            (("checks_primary", _D3),) if with_check_manifest else ()
        ),
        verified=verified,
    )


def contract(
    provider: FixedProvider,
    authority: ProviderAuthorityBinding,
    *criterion_ids: str,
) -> ScorerContract:
    deterministic = provider.declared_kind is ScorerKind.DETERMINISTIC
    return ScorerContract(
        kind=provider.declared_kind,
        scorer_id=provider.provider_id,
        scorer_version=provider.provider_version,
        authority_id=authority.authority_id,
        authority_manifest_digest=authority.authority_manifest_digest,
        criterion_ids=tuple(criterion_ids),
        evidence_action_kinds=(
            (ActionKind.CHECK_RUN,) if deterministic else ()
        ),
        check_set_manifests=authority.check_set_manifests,
        artifact_manifests=authority.artifact_manifests,
        requires_attestation=not deterministic,
    )


class ImportedEvaluationProtocolTests(unittest.TestCase):
    def test_strict_round_trip_is_canonical_and_authority_free(self) -> None:
        observation = imported("criterion_reasoning", "criterion_accuracy")
        decoded = ImportedEvaluation.from_json(
            json.dumps(observation.terms(), separators=(",", ":"))
        )

        self.assertEqual(decoded, observation)
        self.assertEqual(
            [item.criterion_id for item in decoded.criteria],
            ["criterion_accuracy", "criterion_reasoning"],
        )
        criterion_fields = set(decoded.criteria[0].terms())
        self.assertNotIn("scorer_kind", criterion_fields)
        self.assertNotIn("provider_id", criterion_fields)
        self.assertNotIn("authority_id", criterion_fields)

    def test_authority_and_artifact_injection_are_rejected(self) -> None:
        terms = imported("criterion_accuracy").terms()
        for field, value in (
            ("scorer_kind", "deterministic"),
            ("authority_id", "authority.attacker"),
            ("verified", True),
            ("artifact_body", "print('not allowed')"),
        ):
            mutated = json.loads(json.dumps(terms))
            mutated["criteria"][0][field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ScoringProtocolError, "unexpected"):
                    ImportedEvaluation.from_terms(mutated)

    def test_duplicate_and_nonfinite_json_are_rejected(self) -> None:
        duplicate = '{"criteria":[],"schema_version":1,"schema_version":1}'
        with self.assertRaisesRegex(ScoringProtocolError, "duplicate"):
            ImportedEvaluation.from_json(duplicate)

        valid = imported("criterion_accuracy").terms()
        raw = json.dumps(valid).replace('"reliability": 0.9', '"reliability": NaN')
        with self.assertRaisesRegex(ScoringProtocolError, "invalid number"):
            ImportedEvaluation.from_json(raw)
        overflow = json.dumps(valid).replace(
            '"reliability": 0.9', '"reliability": 1e999'
        )
        with self.assertRaisesRegex(ScoringProtocolError, "non-finite"):
            ImportedEvaluation.from_json(overflow)

    def test_bool_scores_and_invalid_status_score_pairs_fail_closed(self) -> None:
        with self.assertRaisesRegex(ScoringProtocolError, "finite"):
            ImportedCriterionResult(
                criterion_id="criterion_accuracy",
                status=EvaluationStatus.VALID,
                score=True,
                outcome_code="observed",
                phase=ActionPhase.UNASSISTED,
            )
        with self.assertRaisesRegex(ScoringProtocolError, "cannot carry"):
            criterion(
                "criterion_accuracy",
                status=EvaluationStatus.MISSING,
                score=0.0,
            )

    def test_request_surface_contains_no_executable_content(self) -> None:
        scoring_request = request("criterion_accuracy")
        self.assertEqual(
            set(scoring_request.terms()),
            {
                "evaluation_id",
                "trace_id",
                "task_id",
                "task_version",
                "task_digest",
                "action_trace_digest",
                "criterion_ids",
                "scorer_contract",
                "scorer_contract_digest",
            },
        )
        serialized = json.dumps(scoring_request.terms())
        for forbidden in ("artifact_body", "command", "callback", "source_code"):
            self.assertNotIn(forbidden, serialized)


class DirectImportAuthorityTests(unittest.TestCase):
    def test_direct_input_cannot_claim_deterministic_or_human_authority(self) -> None:
        scoring_request = request("criterion_accuracy")
        observation = imported("criterion_accuracy", attested=True)
        expectations = {
            ScorerKind.DETERMINISTIC: (
                ScorerKind.IMPORTED,
                "direct_import_cannot_claim_authority",
            ),
            ScorerKind.HUMAN: (
                ScorerKind.IMPORTED,
                "unverified_human_shadow_only",
            ),
            ScorerKind.MODEL: (
                ScorerKind.MODEL,
                "model_score_shadow_only",
            ),
            ScorerKind.IMPORTED: (
                ScorerKind.IMPORTED,
                "imported_score_unadjudicated",
            ),
        }
        for declared_kind, (effective_kind, reason) in expectations.items():
            with self.subTest(declared_kind=declared_kind):
                normalized = normalize_imported_evaluation(
                    scoring_request,
                    observation,
                    provider_id=f"external.{declared_kind.value}",
                    provider_version="v1",
                    declared_kind=declared_kind,
                )
                self.assertIsInstance(normalized.evaluation, TaskEvaluation)
                self.assertIs(
                    normalized.evaluation.criteria[0].scorer_kind,
                    effective_kind,
                )
                self.assertEqual(normalized.decisions[0].reason_code, reason)
                self.assertTrue(normalized.shadow_only)
                self.assertIs(
                    normalized.normalization_mode,
                    NormalizationMode.DIRECT_IMPORT,
                )
                self.assertFalse(normalized.provider.verified)

    def test_normalization_pins_the_entire_task_evaluation_envelope(self) -> None:
        scoring_request = request("criterion_accuracy", "criterion_reasoning")
        normalized = normalize_imported_evaluation(
            scoring_request,
            imported("criterion_reasoning", "criterion_accuracy"),
            provider_id="external.importer",
            provider_version="v7",
        )
        self.assertEqual(
            NormalizedScoringResult.from_terms(normalized.terms()).terms(),
            normalized.terms(),
        )

        evaluation = normalized.evaluation
        self.assertEqual(evaluation.id, scoring_request.evaluation_id)
        self.assertEqual(evaluation.trace_id, scoring_request.trace_id)
        self.assertEqual(evaluation.task_id, scoring_request.task_id)
        self.assertEqual(evaluation.task_version, scoring_request.task_version)
        self.assertEqual(evaluation.task_digest, scoring_request.task_digest)
        self.assertEqual(
            evaluation.action_trace_digest,
            scoring_request.action_trace_digest,
        )
        self.assertEqual(
            [item.criterion_id for item in evaluation.criteria],
            ["criterion_accuracy", "criterion_reasoning"],
        )

    def test_missing_or_extra_criteria_are_not_silently_normalized(self) -> None:
        scoring_request = request("criterion_accuracy", "criterion_reasoning")
        with self.assertRaisesRegex(ScoringProtocolError, "explicitly cover"):
            normalize_imported_evaluation(
                scoring_request,
                imported("criterion_accuracy"),
                provider_id="external.importer",
                provider_version="v1",
            )

    def test_normalized_envelope_rejects_provider_and_decision_forgery(
        self,
    ) -> None:
        normalized = normalize_imported_evaluation(
            request("criterion_accuracy"),
            imported("criterion_accuracy"),
            provider_id="external.importer",
            provider_version="v1",
        )
        with self.assertRaisesRegex(
            ScoringProtocolError, "binding digest does not match"
        ):
            replace(
                normalized.provider,
                provider_id="external.forged",
            )

        forged_decision = CriterionAuthorityDecision(
            criterion_id="criterion_accuracy",
            declared_kind=ScorerKind.IMPORTED,
            effective_kind=ScorerKind.IMPORTED,
            reason_code="forged_authority_reason",
        )
        with self.assertRaisesRegex(
            ScoringProtocolError, "authority does not match"
        ):
            NormalizedScoringResult(
                evaluation=normalized.evaluation,
                request=normalized.request,
                provider=normalized.provider,
                decisions=(forged_decision,),
                normalization_mode=normalized.normalization_mode,
            )


class RegisteredAuthorityTests(unittest.TestCase):
    def test_verified_deterministic_provider_is_manifest_bound(self) -> None:
        observation = imported("criterion_accuracy")
        provider = FixedProvider(
            "checks.python",
            "v3",
            ScorerKind.DETERMINISTIC,
            observation,
        )
        authority = binding(
            provider, verified=True, with_check_manifest=True
        )
        registry = ScoringProviderRegistry()
        summary = registry.register(provider, authority)

        self.assertIsInstance(provider, TaskScoringProvider)
        self.assertEqual(registry.list(), (summary,))
        self.assertEqual(
            registry.inspect("checks.python", "v3").check_set_manifests,
            (("checks_primary", _D3),),
        )
        first = registry.score(
            "checks.python",
            "v3",
            request(
                "criterion_accuracy",
                scorer_contract=contract(
                    provider, authority, "criterion_accuracy"
                ),
            ),
        )
        second = registry.score(
            "checks.python",
            "v3",
            request(
                "criterion_accuracy",
                scorer_contract=contract(
                    provider, authority, "criterion_accuracy"
                ),
            ),
        )
        self.assertIs(
            first.evaluation.criteria[0].scorer_kind,
            ScorerKind.DETERMINISTIC,
        )
        self.assertEqual(
            first.decisions[0].reason_code,
            "verified_deterministic_authority",
        )
        self.assertFalse(first.shadow_only)
        self.assertEqual(first.digest, second.digest)
        self.assertIs(
            first.normalization_mode,
            NormalizationMode.REGISTERED_PROVIDER,
        )
        self.assertEqual(
            first.request.scorer_contract_digest,
            first.terms()["request"]["scorer_contract_digest"],
        )

    def test_verified_provider_requires_the_matching_task_contract(self) -> None:
        provider = FixedProvider(
            "checks.boundary",
            "v1",
            ScorerKind.DETERMINISTIC,
            imported("criterion_accuracy"),
        )
        authority = binding(
            provider, verified=True, with_check_manifest=True
        )
        registry = ScoringProviderRegistry()
        registry.register(provider, authority)
        with self.assertRaisesRegex(
            ScoringProtocolError, "release-pinned scorer contract"
        ):
            registry.score(
                provider.provider_id,
                provider.provider_version,
                request("criterion_accuracy"),
            )

        mismatched = ScorerContract(
            kind=provider.declared_kind,
            scorer_id=provider.provider_id,
            scorer_version=provider.provider_version,
            authority_id=authority.authority_id,
            authority_manifest_digest=_D4,
            criterion_ids=("criterion_accuracy",),
            evidence_action_kinds=(ActionKind.CHECK_RUN,),
            check_set_manifests=authority.check_set_manifests,
        )
        with self.assertRaisesRegex(
            ScoringProtocolError, "provider manifests"
        ):
            registry.score(
                provider.provider_id,
                provider.provider_version,
                request(
                    "criterion_accuracy",
                    scorer_contract=mismatched,
                ),
            )

    def test_human_authority_requires_registry_verification_and_attestation(self) -> None:
        cases = (
            (False, True, ScorerKind.IMPORTED, "unverified_human_shadow_only"),
            (
                True,
                False,
                ScorerKind.IMPORTED,
                "missing_verified_human_attestation",
            ),
            (True, True, ScorerKind.HUMAN, "verified_human_authority"),
        )
        for index, (verified, attested, effective, reason) in enumerate(cases):
            provider = FixedProvider(
                f"human.reviewer-{index}",
                "v1",
                ScorerKind.HUMAN,
                imported("criterion_reasoning", attested=attested),
            )
            registry = ScoringProviderRegistry()
            authority = binding(provider, verified=verified)
            registry.register(provider, authority)
            scorer_contract = (
                contract(provider, authority, "criterion_reasoning")
                if verified
                else None
            )
            normalized = registry.score(
                provider.provider_id,
                "v1",
                request(
                    "criterion_reasoning",
                    scorer_contract=scorer_contract,
                ),
            )
            with self.subTest(verified=verified, attested=attested):
                self.assertIs(
                    normalized.evaluation.criteria[0].scorer_kind, effective
                )
                self.assertEqual(normalized.decisions[0].reason_code, reason)

    def test_model_and_imported_bindings_cannot_be_marked_verified(self) -> None:
        for kind in (ScorerKind.MODEL, ScorerKind.IMPORTED):
            provider = FixedProvider(
                f"shadow.{kind.value}",
                "v1",
                kind,
                imported("criterion_accuracy"),
            )
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(
                    ScoringProtocolError, "cannot receive verified"
                ):
                    binding(provider, verified=True)

    def test_verified_deterministic_binding_requires_closed_manifests(self) -> None:
        provider = FixedProvider(
            "checks.unbound",
            "v1",
            ScorerKind.DETERMINISTIC,
            imported("criterion_accuracy"),
        )
        with self.assertRaisesRegex(
            ScoringProtocolError, "requires a closed"
        ):
            binding(provider, verified=True)


class SyntheticProviderTests(unittest.TestCase):
    def test_synthetic_provider_is_explicit_opt_in_and_always_shadow(self) -> None:
        provider = SyntheticDeterministicProvider(
            imported("criterion_accuracy"),
            check_set_manifests=(("checks_fixture", _D3),),
        )
        with self.assertRaisesRegex(
            ScoringProtocolError, "allow_synthetic=True"
        ):
            ScoringProviderRegistry().register(
                provider, provider.authority_binding
            )

        registry = ScoringProviderRegistry(allow_synthetic=True)
        summary = registry.register(provider, provider.authority_binding)
        normalized = registry.score(
            provider.provider_id,
            provider.provider_version,
            request("criterion_accuracy"),
        )

        self.assertTrue(summary.synthetic)
        self.assertTrue(summary.shadow_only)
        self.assertTrue(summary.provider_id.startswith("synthetic."))
        self.assertIs(
            normalized.evaluation.criteria[0].scorer_kind,
            ScorerKind.IMPORTED,
        )
        self.assertEqual(
            normalized.decisions[0].reason_code,
            "synthetic_provider_shadow_only",
        )
        self.assertTrue(normalized.shadow_only)
        self.assertEqual(
            normalized.digest,
            registry.score(
                provider.provider_id,
                provider.provider_version,
                request("criterion_accuracy"),
            ).digest,
        )

    def test_synthetic_provider_can_never_receive_verified_authority(self) -> None:
        provider = SyntheticDeterministicProvider(
            imported("criterion_accuracy")
        )
        trusted = ProviderAuthorityBinding(
            provider_id=provider.provider_id,
            provider_version=provider.provider_version,
            declared_kind=provider.declared_kind,
            authority_id="authority.synthetic-forbidden",
            authority_manifest_digest=_D2,
            check_set_manifests=(("checks_fixture", _D3),),
            verified=True,
        )
        with self.assertRaisesRegex(
            ScoringProtocolError, "cannot receive verified"
        ):
            ScoringProviderRegistry(allow_synthetic=True).register(
                provider, trusted
            )

    def test_synthetic_fixture_must_exactly_match_request(self) -> None:
        provider = SyntheticDeterministicProvider(
            imported("criterion_accuracy")
        )
        registry = ScoringProviderRegistry(allow_synthetic=True)
        registry.register(provider, provider.authority_binding)
        with self.assertRaisesRegex(ProviderExecutionError, "failed"):
            registry.score(
                provider.provider_id,
                provider.provider_version,
                request("criterion_reasoning"),
            )


class RegistryFailClosedTests(unittest.TestCase):
    def test_registry_never_overwrites_an_exact_provider_version(self) -> None:
        provider = FixedProvider(
            "checks.primary",
            "v1",
            ScorerKind.DETERMINISTIC,
            imported("criterion_accuracy"),
        )
        registry = ScoringProviderRegistry()
        authority = binding(
            provider, verified=True, with_check_manifest=True
        )
        registry.register(provider, authority)
        with self.assertRaisesRegex(ScoringProtocolError, "already registered"):
            registry.register(provider, authority)

    def test_registry_list_is_sorted_and_unknown_provider_is_explicit(self) -> None:
        registry = ScoringProviderRegistry()
        for provider_id in ("provider.zeta", "provider.alpha"):
            provider = FixedProvider(
                provider_id,
                "v1",
                ScorerKind.IMPORTED,
                imported("criterion_accuracy"),
            )
            registry.register(provider, binding(provider, verified=False))
        self.assertEqual(
            [item.provider_id for item in registry.list()],
            ["provider.alpha", "provider.zeta"],
        )
        with self.assertRaises(ProviderNotFoundError):
            registry.inspect("provider.missing", "v1")

    def test_registered_provider_terms_reject_forged_synthetic_namespace(
        self,
    ) -> None:
        provider = FixedProvider(
            "provider.namespace",
            "v1",
            ScorerKind.IMPORTED,
            imported("criterion_accuracy"),
        )
        registry = ScoringProviderRegistry()
        summary = registry.register(
            provider, binding(provider, verified=False)
        )

        forged_non_synthetic = summary.terms()
        forged_non_synthetic["provider_id"] = "synthetic.forged"
        forged_binding = ProviderAuthorityBinding(
            provider_id=forged_non_synthetic["provider_id"],
            provider_version=summary.provider_version,
            declared_kind=summary.declared_kind,
            authority_id=summary.authority_id,
            authority_manifest_digest=summary.authority_manifest_digest,
            check_set_manifests=summary.check_set_manifests,
            artifact_manifests=summary.artifact_manifests,
            verified=summary.verified,
        )
        forged_non_synthetic["binding_digest"] = forged_binding.digest
        with self.assertRaisesRegex(
            ScoringProtocolError, "synthetic namespace"
        ):
            RegisteredProvider.from_terms(forged_non_synthetic)

        forged_synthetic = summary.terms()
        forged_synthetic["synthetic"] = True
        with self.assertRaisesRegex(
            ScoringProtocolError, "Synthetic provider IDs"
        ):
            RegisteredProvider.from_terms(forged_synthetic)

    def test_provider_identity_drift_and_nonprotocol_results_fail_closed(self) -> None:
        provider = FixedProvider(
            "provider.mutable",
            "v1",
            ScorerKind.IMPORTED,
            imported("criterion_accuracy"),
        )
        registry = ScoringProviderRegistry()
        registry.register(provider, binding(provider, verified=False))
        provider.provider_version = "v2"
        with self.assertRaisesRegex(ProviderExecutionError, "identity changed"):
            registry.score(
                "provider.mutable", "v1", request("criterion_accuracy")
            )

        bad_provider = FixedProvider(
            "provider.bad-result",
            "v1",
            ScorerKind.IMPORTED,
            {"criteria": []},
        )
        bad_registry = ScoringProviderRegistry()
        bad_registry.register(
            bad_provider, binding(bad_provider, verified=False)
        )
        with self.assertRaisesRegex(
            ProviderExecutionError, "non-protocol result"
        ):
            bad_registry.score(
                "provider.bad-result", "v1", request("criterion_accuracy")
            )

    def test_provider_identity_cannot_change_during_scoring(self) -> None:
        class SelfMutatingProvider(FixedProvider):
            def score(self, scoring_request):
                result = super().score(scoring_request)
                self.provider_version = "v2"
                return result

        provider = SelfMutatingProvider(
            "provider.self-mutating",
            "v1",
            ScorerKind.IMPORTED,
            imported("criterion_accuracy"),
        )
        registry = ScoringProviderRegistry()
        registry.register(provider, binding(provider, verified=False))

        with self.assertRaisesRegex(
            ProviderExecutionError, "changed while scoring"
        ):
            registry.score(
                "provider.self-mutating",
                "v1",
                request("criterion_accuracy"),
            )

    def test_provider_exceptions_are_wrapped_without_fabricating_a_result(self) -> None:
        provider = FixedProvider(
            "provider.failure",
            "v1",
            ScorerKind.IMPORTED,
            imported("criterion_accuracy"),
        )
        provider.failure = RuntimeError("external scorer unavailable")
        registry = ScoringProviderRegistry()
        registry.register(provider, binding(provider, verified=False))
        with self.assertRaises(ProviderExecutionError) as raised:
            registry.score(
                "provider.failure", "v1", request("criterion_accuracy")
            )
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)


if __name__ == "__main__":
    unittest.main()
