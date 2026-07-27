# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from tsq.artifact_runner import (
    BUNDLED_WORKER_SHA256,
    CAUSAL_MASK_ARTIFACT_MANIFEST_DIGEST,
    CAUSAL_MASK_CHECK_SET_MANIFEST_DIGEST,
    MAX_RUNNER_ARTIFACT_BYTES,
    ArtifactCheckerId,
    ArtifactProcessReceipt,
    ArtifactResultCode,
    ArtifactRunOutcome,
    ArtifactRunRequest,
    ArtifactRunResult,
    ArtifactRunnerNotFoundError,
    ArtifactRunnerProtocolError,
    SyntheticArtifactRunnerRegistry,
    _invoke_worker,
    build_artifact_run_request,
    bundled_synthetic_binding,
)
from tsq.evidence import canonical_json


def artifact(mask: list[list[bool]]) -> bytes:
    return json.dumps(
        {"mask": mask, "schema_version": 1},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class ArtifactRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.binding = bundled_synthetic_binding()
        self.registry = SyntheticArtifactRunnerRegistry(allow_synthetic=True)
        self.registry.register(self.binding)

    def run_bytes(
        self,
        material: bytes,
        *,
        run_id: str = "run_fixture",
    ) -> ArtifactProcessReceipt:
        request = build_artifact_run_request(
            run_id,
            material,
            self.binding,
        )
        return self.registry.run(request, material)

    def test_correct_and_incorrect_causal_matrices_are_checked_as_data(
        self,
    ) -> None:
        correct = self.run_bytes(
            artifact(
                [
                    [True, False, False],
                    [True, True, False],
                    [True, True, True],
                ]
            )
        )
        self.assertIs(correct.result.outcome, ArtifactRunOutcome.COMPLETED)
        self.assertEqual(
            correct.result.outcome_codes,
            (
                ArtifactResultCode.CAUSAL_VISIBILITY_VALID,
                ArtifactResultCode.MATRIX_SHAPE_VALID,
            ),
        )
        self.assertEqual(
            (
                correct.result.passed,
                correct.result.failed,
                correct.result.errored,
                correct.result.skipped,
            ),
            (2, 0, 0, 0),
        )

        incorrect = self.run_bytes(
            artifact(
                [
                    [True, True, False],
                    [True, True, False],
                    [True, True, True],
                ]
            ),
            run_id="run_incorrect",
        )
        self.assertIs(incorrect.result.outcome, ArtifactRunOutcome.COMPLETED)
        self.assertEqual(
            incorrect.result.outcome_codes,
            (
                ArtifactResultCode.CAUSAL_VISIBILITY_INVALID,
                ArtifactResultCode.MATRIX_SHAPE_VALID,
            ),
        )
        self.assertEqual(
            (
                incorrect.result.passed,
                incorrect.result.failed,
                incorrect.result.errored,
            ),
            (1, 1, 0),
        )

    def test_process_receipt_states_every_non_authority_boundary(self) -> None:
        receipt = self.run_bytes(artifact([[True]]))
        terms = receipt.terms()

        self.assertTrue(terms["synthetic"])
        self.assertTrue(terms["process_separated"])
        self.assertTrue(terms["worker_process_started"])
        self.assertTrue(terms["trusted_checker_executed"])
        for field_name in (
            "operating_system_sandboxed",
            "filesystem_isolation_enforced",
            "network_isolation_enforced",
            "artifact_executed",
            "evaluation_created",
            "learner_projection_applied",
            "mastery_applied",
            "certification_applied",
            "skill_authority",
        ):
            self.assertFalse(terms[field_name])

    def test_binding_request_result_and_process_receipt_are_canonical(
        self,
    ) -> None:
        material = artifact([[True, False], [True, True]])
        request = build_artifact_run_request(
            "run_round_trip",
            material,
            self.binding,
        )
        receipt = self.registry.run(request, material)

        decoded_binding = type(self.binding).from_terms(self.binding.terms())
        decoded_request = ArtifactRunRequest.from_terms(request.terms())
        decoded_result = ArtifactRunResult.from_json(
            canonical_json(receipt.result.terms())
        )
        decoded_receipt = ArtifactProcessReceipt.from_json(
            canonical_json(receipt.terms())
        )
        self.assertEqual(decoded_binding, self.binding)
        self.assertEqual(decoded_request, request)
        self.assertEqual(decoded_result, receipt.result)
        self.assertEqual(decoded_receipt, receipt)
        self.assertEqual(decoded_binding.digest, self.binding.digest)
        self.assertEqual(decoded_request.digest, request.digest)
        self.assertEqual(decoded_result.digest, receipt.result.digest)
        self.assertEqual(decoded_receipt.digest, receipt.digest)
        self.assertEqual(receipt.terms()["binding_digest"], self.binding.digest)
        self.assertEqual(receipt.terms()["request_digest"], request.digest)
        self.assertEqual(
            receipt.terms()["result_digest"],
            receipt.result.digest,
        )

    def test_canonical_decoders_reject_extra_duplicate_and_nonfinite_fields(
        self,
    ) -> None:
        request_terms = build_artifact_run_request(
            "run_strict",
            artifact([[True]]),
            self.binding,
        ).terms()
        request_terms["unexpected"] = True
        with self.assertRaisesRegex(
            ArtifactRunnerProtocolError,
            "unexpected unexpected",
        ):
            ArtifactRunRequest.from_terms(request_terms)

        duplicate_result = (
            '{"artifact_sha256":"'
            + ("0" * 64)
            + '","artifact_sha256":"'
            + ("0" * 64)
            + '","checker_id":"synthetic.causal-mask-matrix",'
            '"checker_version":"v1","errored":0,"failed":0,'
            '"outcome":"completed","outcome_codes":[],'
            '"passed":0,"schema_version":1,"skipped":0}'
        )
        with self.assertRaisesRegex(
            ArtifactRunnerProtocolError,
            "duplicate field",
        ):
            ArtifactRunResult.from_json(duplicate_result)

        with self.assertRaisesRegex(
            ArtifactRunnerProtocolError,
            "non-finite",
        ):
            ArtifactRunResult.from_json('{"value":NaN}')
        with self.assertRaisesRegex(
            ArtifactRunnerProtocolError,
            "floating-point",
        ):
            ArtifactRunResult.from_json('{"value":1.5}')

    def test_hostile_artifact_documents_fail_closed_with_stable_codes(
        self,
    ) -> None:
        cases = (
            (
                b'{"mask":[[true]],"schema_version":1,'
                b'"schema_version":1}',
                ArtifactResultCode.DUPLICATE_FIELD,
            ),
            (
                b'{"mask":[[true]],"schema_version":NaN}',
                ArtifactResultCode.NONFINITE_NUMBER,
            ),
            (
                b'{"mask":[[true]],"schema_version":1,"extra":false}',
                ArtifactResultCode.UNKNOWN_OR_MISSING_FIELDS,
            ),
            (
                b'{"mask":[[true]],"schema_version":2}',
                ArtifactResultCode.INVALID_SCHEMA_VERSION,
            ),
            (
                b'{"mask":[[true,false],[true]],"schema_version":1}',
                ArtifactResultCode.INVALID_MATRIX,
            ),
            (
                b'[{"mask":[[true]],"schema_version":1}]',
                ArtifactResultCode.INVALID_DOCUMENT,
            ),
            (
                b'{"mask":[[true]],"schema_version":1.0}',
                ArtifactResultCode.INVALID_JSON,
            ),
            (
                b"\xff",
                ArtifactResultCode.INVALID_UTF8,
            ),
            (
                (
                    b'{"mask":[[true]],"schema_version":1,"nested":'
                    + (b"[" * 9)
                    + b"0"
                    + (b"]" * 9)
                    + b"}"
                ),
                ArtifactResultCode.JSON_DEPTH_EXCEEDED,
            ),
        )
        for index, (material, expected_code) in enumerate(cases):
            with self.subTest(expected_code=expected_code):
                receipt = self.run_bytes(
                    material,
                    run_id=f"run_hostile_{index}",
                )
                self.assertIs(
                    receipt.result.outcome,
                    ArtifactRunOutcome.INVALID_ARTIFACT,
                )
                self.assertEqual(
                    receipt.result.outcome_codes,
                    (expected_code,),
                )
                self.assertEqual(receipt.result.errored, 1)
                self.assertTrue(receipt.worker_process_started)

    def test_empty_and_oversized_artifacts_are_rejected_before_process_start(
        self,
    ) -> None:
        empty = self.run_bytes(b"", run_id="run_empty")
        self.assertIs(
            empty.result.outcome,
            ArtifactRunOutcome.INVALID_ARTIFACT,
        )
        self.assertEqual(
            empty.result.outcome_codes,
            (ArtifactResultCode.EMPTY_INPUT,),
        )
        self.assertFalse(empty.worker_process_started)

        oversized_material = b"x" * (MAX_RUNNER_ARTIFACT_BYTES + 1)
        oversized = self.run_bytes(
            oversized_material,
            run_id="run_oversized",
        )
        self.assertIs(
            oversized.result.outcome,
            ArtifactRunOutcome.INVALID_ARTIFACT,
        )
        self.assertEqual(
            oversized.result.outcome_codes,
            (ArtifactResultCode.INPUT_TOO_LARGE,),
        )
        self.assertFalse(oversized.worker_process_started)

    def test_request_must_match_exact_immutable_byte_snapshot(self) -> None:
        first = artifact([[True]])
        second = b'{"schema_version":1,"mask":[[true]]}'
        self.assertEqual(len(first), len(second))
        request = build_artifact_run_request(
            "run_snapshot",
            first,
            self.binding,
        )
        with self.assertRaisesRegex(
            ArtifactRunnerProtocolError,
            "digest",
        ):
            self.registry.run(request, second)
        with self.assertRaisesRegex(
            ArtifactRunnerProtocolError,
            "length",
        ):
            self.registry.run(replace(request, artifact_size_bytes=0), first)
        with self.assertRaisesRegex(
            ArtifactRunnerProtocolError,
            "immutable bytes",
        ):
            self.registry.run(request, bytearray(first))  # type: ignore[arg-type]

    def test_registry_is_explicit_synthetic_only_and_has_no_callback_port(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ArtifactRunnerProtocolError,
            "allow_synthetic=True",
        ):
            SyntheticArtifactRunnerRegistry().register(self.binding)
        with self.assertRaisesRegex(
            ArtifactRunnerProtocolError,
            "exact bundled",
        ):
            SyntheticArtifactRunnerRegistry(
                allow_synthetic=True
            ).register(replace(self.binding, timeout_ms=100))
        with self.assertRaisesRegex(
            ArtifactRunnerProtocolError,
            "already registered",
        ):
            self.registry.register(self.binding)
        with self.assertRaises(ArtifactRunnerNotFoundError):
            self.registry.inspect(
                ArtifactCheckerId.CAUSAL_MASK_MATRIX,
                "v999",
            )
        self.assertEqual(self.registry.list(), (self.binding,))
        self.assertNotIn("adapter", self.registry.register.__code__.co_varnames)
        self.assertNotIn("callback", self.registry.register.__code__.co_varnames)

    def test_frozen_worker_digest_and_manifests_are_explicit_commitments(
        self,
    ) -> None:
        worker_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "tsq"
            / "_causal_mask_checker.py"
        )
        self.assertEqual(
            hashlib.sha256(worker_path.read_bytes()).hexdigest(),
            BUNDLED_WORKER_SHA256,
        )
        self.assertEqual(self.binding.worker_sha256, BUNDLED_WORKER_SHA256)
        self.assertRegex(CAUSAL_MASK_ARTIFACT_MANIFEST_DIGEST, r"^[0-9a-f]{64}$")
        self.assertRegex(CAUSAL_MASK_CHECK_SET_MANIFEST_DIGEST, r"^[0-9a-f]{64}$")
        self.assertEqual(
            self.binding.artifact_manifest_digest,
            CAUSAL_MASK_ARTIFACT_MANIFEST_DIGEST,
        )
        self.assertEqual(
            self.binding.check_set_manifest_digest,
            CAUSAL_MASK_CHECK_SET_MANIFEST_DIGEST,
        )
        with patch(
            "tsq.artifact_runner._read_bundled_worker_source",
            return_value=b"# modified trusted worker\n",
        ):
            with self.assertRaisesRegex(
                ArtifactRunnerProtocolError,
                "frozen v1 digest",
            ):
                bundled_synthetic_binding()

    def test_subprocess_uses_fixed_isolated_mode_private_cwd_and_minimal_env(
        self,
    ) -> None:
        real_popen = subprocess.Popen
        captured: list[tuple[tuple, dict]] = []

        def observing_popen(*arguments, **keywords):
            captured.append((arguments, keywords))
            return real_popen(*arguments, **keywords)

        material = artifact([[True]])
        with patch(
            "tsq.artifact_runner.subprocess.Popen",
            side_effect=observing_popen,
        ):
            receipt = self.run_bytes(material, run_id="run_process_boundary")
        self.assertIs(receipt.result.outcome, ArtifactRunOutcome.COMPLETED)
        self.assertEqual(len(captured), 1)
        positional, keywords = captured[0]
        command = positional[0]
        self.assertEqual(command[0], Path(command[0]).as_posix())
        self.assertEqual(command[1:3], ["-I", "-S"])
        self.assertEqual(Path(command[3]).name, "worker.py")
        private_cwd = Path(keywords["cwd"])
        self.assertEqual(Path(command[3]).parent, private_cwd)
        self.assertFalse(private_cwd.exists())
        self.assertEqual(keywords["env"], {})
        self.assertTrue(keywords["close_fds"])
        self.assertFalse(keywords["shell"])
        self.assertTrue(keywords["start_new_session"])
        self.assertIs(keywords["stdin"], subprocess.PIPE)
        self.assertIs(keywords["stdout"], subprocess.PIPE)
        self.assertIs(keywords["stderr"], subprocess.PIPE)
        self.assertNotIn(material.decode("utf-8"), repr(command))
        self.assertNotIn(material.decode("utf-8"), repr(keywords["env"]))

    def test_learner_strings_are_not_commands_imports_or_persisted_output(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "must-not-exist"
            command_like = (
                "__import__('pathlib').Path("
                + repr(str(marker))
                + ").write_text('executed')"
            )
            material = json.dumps(
                {
                    "mask": [[True]],
                    "schema_version": 1,
                    "payload": command_like,
                },
                separators=(",", ":"),
            ).encode("utf-8")
            receipt = self.run_bytes(material, run_id="run_inert_string")
            self.assertFalse(marker.exists())
            self.assertEqual(
                receipt.result.outcome_codes,
                (ArtifactResultCode.UNKNOWN_OR_MISSING_FIELDS,),
            )
            rendered = canonical_json(receipt.terms())
            self.assertNotIn(command_like, rendered)
            self.assertNotIn(str(marker), rendered)

    def test_host_supervisor_bounds_timeout_and_combined_output(self) -> None:
        material = b"x" * MAX_RUNNER_ARTIFACT_BYTES

        sleeping_source = b"import time\ntime.sleep(5)\n"
        sleeping_binding = replace(
            self.binding,
            worker_sha256=hashlib.sha256(sleeping_source).hexdigest(),
            timeout_ms=50,
        )
        sleeping_request = build_artifact_run_request(
            "run_timeout",
            material,
            sleeping_binding,
        )
        timed = _invoke_worker(
            sleeping_request,
            sleeping_binding,
            material,
            sleeping_source,
        )
        self.assertTrue(timed.process_started)
        self.assertIs(timed.result.outcome, ArtifactRunOutcome.TIMED_OUT)
        self.assertEqual(
            timed.result.outcome_codes,
            (ArtifactResultCode.WORKER_TIMEOUT,),
        )

        noisy_source = (
            b"import sys\n"
            b"sys.stdin.buffer.read()\n"
            b"sys.stdout.buffer.write(b'x' * 20000)\n"
            b"sys.stdout.buffer.flush()\n"
        )
        noisy_binding = replace(
            self.binding,
            worker_sha256=hashlib.sha256(noisy_source).hexdigest(),
            maximum_output_bytes=1_024,
        )
        noisy_request = build_artifact_run_request(
            "run_output_limit",
            material,
            noisy_binding,
        )
        noisy = _invoke_worker(
            noisy_request,
            noisy_binding,
            material,
            noisy_source,
        )
        self.assertTrue(noisy.process_started)
        self.assertIs(noisy.result.outcome, ArtifactRunOutcome.PROTOCOL_ERROR)
        self.assertEqual(
            noisy.result.outcome_codes,
            (ArtifactResultCode.WORKER_OUTPUT_LIMIT,),
        )

    @unittest.skipUnless(hasattr(__import__("os"), "fork"), "requires POSIX fork")
    def test_timeout_kills_descendants_holding_worker_pipes(self) -> None:
        import time

        material = artifact([[True]])
        descendant_source = (
            b"import os, time\n"
            b"child = os.fork()\n"
            b"if child == 0:\n"
            b"    time.sleep(5)\n"
            b"    os._exit(0)\n"
            b"os._exit(0)\n"
        )
        binding = replace(
            self.binding,
            worker_sha256=hashlib.sha256(descendant_source).hexdigest(),
            timeout_ms=50,
        )
        request = build_artifact_run_request(
            "run_descendant_timeout",
            material,
            binding,
        )
        started = time.monotonic()
        invocation = _invoke_worker(
            request,
            binding,
            material,
            descendant_source,
        )
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 1.0)
        self.assertTrue(invocation.process_started)
        self.assertIs(invocation.result.outcome, ArtifactRunOutcome.TIMED_OUT)

    def test_stderr_nonzero_and_invalid_protocol_are_closed_failures(self) -> None:
        material = artifact([[True]])
        cases = (
            (
                b"import sys\nsys.stderr.write('private failure')\n",
                ArtifactRunOutcome.PROTOCOL_ERROR,
                ArtifactResultCode.WORKER_STDERR,
            ),
            (
                b"raise SystemExit(7)\n",
                ArtifactRunOutcome.WORKER_FAILED,
                ArtifactResultCode.WORKER_EXIT_NONZERO,
            ),
            (
                b"print('{}')\n",
                ArtifactRunOutcome.PROTOCOL_ERROR,
                ArtifactResultCode.WORKER_PROTOCOL_INVALID,
            ),
        )
        for index, (source, outcome, code) in enumerate(cases):
            with self.subTest(code=code):
                binding = replace(
                    self.binding,
                    worker_sha256=hashlib.sha256(source).hexdigest(),
                )
                request = build_artifact_run_request(
                    f"run_worker_failure_{index}",
                    material,
                    binding,
                )
                invocation = _invoke_worker(
                    request,
                    binding,
                    material,
                    source,
                )
                self.assertTrue(invocation.process_started)
                self.assertIs(invocation.result.outcome, outcome)
                self.assertEqual(invocation.result.outcome_codes, (code,))
                self.assertNotIn(
                    "private failure",
                    canonical_json(invocation.result.terms()),
                )

    def test_process_start_failure_is_bounded_without_claiming_execution(
        self,
    ) -> None:
        material = artifact([[True]])
        request = build_artifact_run_request(
            "run_start_failure",
            material,
            self.binding,
        )
        with patch(
            "tsq.artifact_runner.subprocess.Popen",
            side_effect=OSError("private launch detail"),
        ):
            receipt = self.registry.run(request, material)
        self.assertFalse(receipt.worker_process_started)
        self.assertIs(receipt.result.outcome, ArtifactRunOutcome.WORKER_FAILED)
        self.assertEqual(
            receipt.result.outcome_codes,
            (ArtifactResultCode.WORKER_START_FAILED,),
        )
        self.assertNotIn("private launch detail", canonical_json(receipt.terms()))

    def test_result_and_receipt_reject_forged_boundaries(self) -> None:
        with self.assertRaisesRegex(
            ArtifactRunnerProtocolError,
            "inconsistent",
        ):
            ArtifactRunResult(
                checker_id=ArtifactCheckerId.CAUSAL_MASK_MATRIX,
                checker_version="v1",
                artifact_sha256="0" * 64,
                outcome=ArtifactRunOutcome.COMPLETED,
                outcome_codes=(ArtifactResultCode.CAUSAL_VISIBILITY_VALID,),
                passed=1,
                failed=0,
                errored=0,
                skipped=0,
            )

        receipt = self.run_bytes(artifact([[True]]), run_id="run_tamper")
        with self.assertRaisesRegex(
            ArtifactRunnerProtocolError,
            "process state",
        ):
            ArtifactProcessReceipt(
                request=receipt.request,
                binding=receipt.binding,
                result=receipt.result,
                worker_process_started=False,
            )

        forged = receipt.terms()
        forged["operating_system_sandboxed"] = True
        with self.assertRaisesRegex(
            ArtifactRunnerProtocolError,
            "must be False",
        ):
            ArtifactProcessReceipt.from_terms(forged)

        forged = receipt.terms()
        forged["result_digest"] = "0" * 64
        with self.assertRaisesRegex(
            ArtifactRunnerProtocolError,
            "mismatched digest",
        ):
            ArtifactProcessReceipt.from_terms(forged)

        with self.assertRaisesRegex(
            ArtifactRunnerProtocolError,
            "inconsistent",
        ):
            ArtifactRunResult(
                checker_id=ArtifactCheckerId.CAUSAL_MASK_MATRIX,
                checker_version="v1",
                artifact_sha256="0" * 64,
                outcome=ArtifactRunOutcome.INVALID_ARTIFACT,
                outcome_codes=(
                    ArtifactResultCode.ARTIFACT_DIGEST_MISMATCH,
                ),
                passed=0,
                failed=0,
                errored=1,
                skipped=0,
            )


if __name__ == "__main__":
    unittest.main()
