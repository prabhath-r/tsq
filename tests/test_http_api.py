# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from tsq.corpus import read_and_parse
from tsq.http_api import (
    DEFAULT_ALLOWED_ORIGINS,
    ApiApplication,
    build_parser,
    main,
    prepare_database,
)
from tsq.store import Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"


def json_body(payload: dict) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


class HttpApiTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.tempdir.name) / "api.db")
        self.database.initialize()
        self.database.import_corpus(
            *read_and_parse(CORPUS, include_catalog=True)
        )
        self.app = ApiApplication(self.database)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        *,
        headers: dict[str, str] | None = None,
    ):
        request_headers = dict(headers or {})
        body = b""
        if payload is not None:
            request_headers.setdefault("Content-Type", "application/json")
            body = json_body(payload)
        return self.app.dispatch(
            method,
            path,
            headers=request_headers,
            body=body,
        )

    def ensure_learner(self, learner_id: str = "web-learner") -> dict:
        response = self.request(
            "POST", "/api/v1/learners", {"learner_id": learner_id}
        )
        self.assertEqual(response.status, 200, response.payload)
        return response.payload

    def start_rag_session(self, learner_id: str = "web-learner") -> dict:
        response = self.request(
            "POST",
            "/api/v1/sessions",
            {
                "learner_id": learner_id,
                "topic_id": "t_retrieval_augmented_generation",
                "mode": "learn",
            },
            headers={"Idempotency-Key": f"start:{learner_id}:rag"},
        )
        self.assertEqual(response.status, 201, response.payload)
        return response.payload

    def test_health_and_catalog_are_the_active_release(self) -> None:
        health = self.request("GET", "/health")
        self.assertEqual(health.status, 200)
        self.assertEqual(health.payload["status"], "ok")
        self.assertEqual(health.payload["corpus"]["questions"], 532)
        self.assertEqual(health.payload["corpus"]["active_questions"], 491)
        self.assertEqual(health.payload["corpus"]["retired_questions"], 41)
        self.assertEqual(health.payload["corpus"]["active_families"], 269)
        self.assertEqual(health.payload["corpus"]["misconceptions"], 186)
        self.assertEqual(health.payload["corpus"]["sources"], 47)
        self.assertEqual(health.payload["corpus"]["topics"], 16)

        catalog = self.request("GET", "/api/v1/catalog")
        self.assertEqual(catalog.status, 200)
        self.assertEqual(
            catalog.payload["release_id"], health.payload["corpus_release_id"]
        )
        self.assertEqual(len(catalog.payload["topics"]), 16)
        rag = next(
            topic
            for topic in catalog.payload["topics"]
            if topic["id"] == "t_retrieval_augmented_generation"
        )
        self.assertEqual(
            rag["path"][-1], "Retrieval-Augmented Generation"
        )
        self.assertGreater(rag["scope_primary_questions"], 0)
        self.assertGreater(rag["scope_learning_objectives"], 0)
        self.assertIn("direct_learning_objectives", rag)

        topics = self.request("GET", "/api/v1/topics")
        self.assertEqual(topics.status, 200)
        self.assertEqual(topics.payload["topics"], catalog.payload["topics"])

    def test_full_rag_journey_uses_engine_idempotency_and_reports(self) -> None:
        learner = self.ensure_learner()
        repeated_learner = self.ensure_learner()
        self.assertEqual(repeated_learner["id"], learner["id"])

        session = self.start_rag_session()
        repeated_start = self.start_rag_session()
        self.assertEqual(repeated_start["id"], session["id"])
        self.assertEqual(
            session["topic_id"], "t_retrieval_augmented_generation"
        )

        selected = self.request(
            "POST",
            f"/api/v1/sessions/{session['id']}/next",
            {"idempotency_key": "next:rag:1"},
        )
        self.assertEqual(selected.status, 200, selected.payload)
        self.assertEqual(
            set(selected.payload),
            {
                "decision_id",
                "session_id",
                "phase",
                "question_id",
                "family_id",
                "kind",
                "pedagogical_role",
                "stem",
                "options",
                "selection",
                "learning_objective",
            },
        )
        self.assertEqual(len(selected.payload["options"]), 4)
        for option in selected.payload["options"]:
            self.assertEqual(set(option), {"id", "text"})
        self.assertNotIn("correct_option_id", selected.payload)
        self.assertNotIn("source_ids", selected.payload)
        self.assertNotIn("question_version", selected.payload)

        repeated_next = self.request(
            "POST",
            f"/api/v1/sessions/{session['id']}/next",
            headers={"Idempotency-Key": "next:rag:1"},
        )
        self.assertEqual(
            repeated_next.payload["decision_id"], selected.payload["decision_id"]
        )

        question = self.database.get_question(selected.payload["question_id"])
        correct_option = next(option for option in question.options if option.correct)
        answered = self.request(
            "POST",
            f"/api/v1/decisions/{selected.payload['decision_id']}/answers",
            {
                "option_id": correct_option.id,
                "confidence": 0.8,
                "response_ms": 0,
                "hint_count": 0,
            },
            headers={"Idempotency-Key": "answer:rag:1"},
        )
        self.assertEqual(answered.status, 200, answered.payload)
        self.assertTrue(answered.payload["correct"])
        self.assertEqual(
            answered.payload["correct_option_id"], correct_option.id
        )
        self.assertTrue(answered.payload["correct_rationale"])

        feedback = self.request(
            "POST",
            f"/api/v1/decisions/{selected.payload['decision_id']}/feedback",
            headers={"Idempotency-Key": "feedback:rag:1"},
        )
        self.assertEqual(feedback.status, 200, feedback.payload)
        self.assertEqual(feedback.payload["action_type"], "feedback_shown")
        repeated_feedback = self.request(
            "POST",
            f"/api/v1/decisions/{selected.payload['decision_id']}/feedback",
            headers={"Idempotency-Key": "feedback:rag:1"},
        )
        self.assertEqual(repeated_feedback.status, 200)
        self.assertTrue(repeated_feedback.payload["idempotent_replay"])

        repeated_answer = self.request(
            "POST",
            f"/api/v1/decisions/{selected.payload['decision_id']}/answers",
            {
                "option_id": correct_option.id,
                "confidence": 0.8,
                "response_ms": 0,
                "hint_count": 0,
            },
            headers={"Idempotency-Key": "answer:rag:1"},
        )
        self.assertEqual(repeated_answer.status, 200)
        self.assertTrue(repeated_answer.payload["idempotent_replay"])

        sessions = self.request(
            "GET", "/api/v1/sessions?learner_id=web-learner"
        )
        self.assertEqual(sessions.status, 200)
        self.assertEqual(sessions.payload["sessions"][0]["questions_answered"], 1)

        fetched = self.request("GET", f"/api/v1/sessions/{session['id']}")
        profile = self.request("GET", "/api/v1/learners/web-learner/profile")
        report = self.request(
            "GET", f"/api/v1/sessions/{session['id']}/report"
        )
        trace = self.request(
            "GET", f"/api/v1/sessions/{session['id']}/trace"
        )
        self.assertEqual(fetched.payload["id"], session["id"])
        self.assertEqual(profile.status, 200, profile.payload)
        self.assertEqual(report.payload["session_id"], session["id"])
        self.assertEqual(trace.payload[0]["id"], selected.payload["decision_id"])

        ended = self.request(
            "POST",
            f"/api/v1/sessions/{session['id']}/end",
            {"status": "completed", "reason": "web_session_complete"},
            headers={"Idempotency-Key": "end:rag:1"},
        )
        self.assertEqual(ended.status, 200, ended.payload)
        self.assertEqual(ended.payload["status"], "completed")
        repeated_end = self.request(
            "POST",
            f"/api/v1/sessions/{session['id']}/end",
            {"status": "completed", "reason": "web_session_complete"},
            headers={"Idempotency-Key": "end:rag:1"},
        )
        self.assertEqual(repeated_end.payload["status"], "completed")

    def test_skip_and_validation_errors_are_explicit(self) -> None:
        self.ensure_learner("skip-learner")
        session = self.start_rag_session("skip-learner")
        selected = self.request(
            "POST", f"/api/v1/sessions/{session['id']}/next"
        )
        skipped = self.request(
            "POST",
            f"/api/v1/decisions/{selected.payload['decision_id']}/answers",
            {"option_id": None, "response_ms": 0, "hint_count": 1},
            headers={"Idempotency-Key": "answer:skip:1"},
        )
        self.assertEqual(skipped.status, 200, skipped.payload)
        self.assertEqual(skipped.payload["outcome"], "abstained")
        self.assertIsNone(skipped.payload["selected_option_id"])

        missing_key = self.request(
            "POST",
            "/api/v1/sessions",
            {
                "learner_id": "skip-learner",
                "topic_id": "t_transformers",
            },
        )
        self.assertEqual(missing_key.status, 400)
        self.assertEqual(
            missing_key.payload["error"]["code"], "validation_error"
        )
        bad_method = self.request(
            "GET", f"/api/v1/sessions/{session['id']}/next"
        )
        self.assertEqual(bad_method.status, 405)
        self.assertEqual(bad_method.headers["Allow"], "POST")
        malformed = self.app.dispatch(
            "POST",
            "/api/v1/learners",
            headers={"Content-Type": "application/json"},
            body=b"{not-json",
        )
        self.assertEqual(malformed.status, 400)

    def test_feedback_receipt_requires_a_committed_answer(self) -> None:
        self.ensure_learner("feedback-learner")
        session = self.start_rag_session("feedback-learner")
        selected = self.request(
            "POST", f"/api/v1/sessions/{session['id']}/next"
        )
        premature = self.request(
            "POST",
            f"/api/v1/decisions/{selected.payload['decision_id']}/feedback",
            headers={"Idempotency-Key": "feedback:premature"},
        )
        self.assertEqual(premature.status, 409)
        self.assertEqual(premature.payload["error"]["code"], "conflict")

        unsafe_answer = self.request(
            "POST",
            f"/api/v1/decisions/{selected.payload['decision_id']}/answers",
            {"option_id": None, "feedback_shown": True},
            headers={"Idempotency-Key": "answer:unsafe-feedback"},
        )
        self.assertEqual(unsafe_answer.status, 400)
        self.assertIn(
            "feedback_shown", unsafe_answer.payload["error"]["message"]
        )

    def test_cors_allows_only_exact_configured_web_origins(self) -> None:
        allowed = self.request(
            "GET",
            "/api/v1/health",
            headers={"Origin": DEFAULT_ALLOWED_ORIGINS[0]},
        )
        self.assertEqual(allowed.status, 200)
        self.assertEqual(
            allowed.headers["Access-Control-Allow-Origin"],
            DEFAULT_ALLOWED_ORIGINS[0],
        )
        rejected = self.request(
            "GET",
            "/api/v1/health",
            headers={"Origin": "https://example.invalid"},
        )
        self.assertEqual(rejected.status, 403)
        self.assertNotIn("Access-Control-Allow-Origin", rejected.headers)
        with self.assertRaisesRegex(ValueError, "remote browser origins"):
            ApiApplication(
                self.database,
                allowed_origins=("https://example.invalid",),
            )
        preflight = self.request(
            "OPTIONS",
            "/api/v1/sessions",
            headers={"Origin": DEFAULT_ALLOWED_ORIGINS[1]},
        )
        self.assertEqual(preflight.status, 204)
        self.assertIn("Idempotency-Key", preflight.headers["Access-Control-Allow-Headers"])

    def test_prepare_database_installs_bundled_corpus_once(self) -> None:
        path = Path(self.tempdir.name) / "starter.db"
        first = prepare_database(path)
        release_id = first.get_active_release_id()
        second = prepare_database(path)
        self.assertEqual(second.get_active_release_id(), release_id)
        self.assertEqual(len(second.get_catalog()["topics"]), 16)

    def test_parser_defaults_to_loopback(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 8765)
        error = io.StringIO()
        with redirect_stderr(error):
            code = main(["--host", "0.0.0.0"])
        self.assertEqual(code, 2)
        self.assertIn("only permits --host 127.0.0.1", error.getvalue())


if __name__ == "__main__":
    unittest.main()
