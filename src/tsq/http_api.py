# SPDX-License-Identifier: MPL-2.0

"""Loopback-only JSON API for the TSQ adaptive-learning engine.

The API is intentionally thin.  It translates HTTP requests into the same
``Database`` and ``AdaptiveEngine`` operations used by the CLI; question
selection, response inference, evidence updates, reports, and traces remain
owned by the engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, unquote, urlsplit

from .cli import (
    _decision_objective_names,
    _ensure_starter_corpus,
    _submission_dict,
)
from .corpus import read_and_parse
from .engine import AdaptiveEngine
from .errors import (
    ConflictError,
    ExhaustedError,
    NotFoundError,
    TSQError,
    ValidationError,
)
from .models import Presentation, SubmissionResult
from .store import SCHEMA_VERSION, Database


API_VERSION = "v1"
API_PREFIX = ("api", API_VERSION)
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://[::1]:3000",
)
MAX_REQUEST_BYTES = 1_048_576
MAX_SESSION_LIST_LIMIT = 200
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ApiResponse:
    """One JSON response returned by :class:`ApiApplication`."""

    status: int
    payload: dict[str, Any] | list[Any] | None = None
    headers: dict[str, str] = field(default_factory=dict)


class MethodNotAllowedError(Exception):
    """Internal routing signal carrying the exact allowed methods."""

    def __init__(self, method: str, allowed: Sequence[str]) -> None:
        self.method = method
        self.allowed = tuple(allowed)
        super().__init__(
            f"Method {method} is not allowed; use {', '.join(self.allowed)}."
        )


def presentation_payload(presentation: Presentation) -> dict[str, Any]:
    """Serialize a pending decision without exposing its answer key."""

    question = presentation.question
    payload: dict[str, Any] = {
        "decision_id": presentation.decision_id,
        "session_id": presentation.session_id,
        "phase": presentation.phase.value,
        "question_id": question.id,
        "family_id": question.family_id,
        "kind": question.kind.value,
        "pedagogical_role": presentation.pedagogical_role,
        "stem": question.stem,
        "options": [
            {"id": option.id, "text": option.text}
            for option in presentation.ordered_options
        ],
        "selection": {
            "rationale": presentation.rationale,
            "propensity": presentation.propensity,
            "score": presentation.score.terms(),
        },
    }
    if question.objective is not None:
        objective = question.objective
        payload["learning_objective"] = {
            "id": objective.id,
            "name": objective.name,
            "description": objective.description,
            "operation": objective.operation.value,
            "evidence_type": objective.evidence_type,
        }
    return payload


def submission_payload(
    result: SubmissionResult,
    objective_names: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Use the CLI's exact post-answer contract for interface parity."""

    return _submission_dict(result, objective_names)


def _validate_origin(origin: str) -> str:
    try:
        parsed = urlsplit(origin)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"Invalid allowed origin: {origin!r}.") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Allowed origins must be exact HTTP(S) origins without a path, "
            f"query, credentials, or fragment: {origin!r}."
        )
    hostname = parsed.hostname.casefold()
    if hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError(
            "Allowed origins must use localhost, 127.0.0.1, or ::1; "
            f"remote browser origins cannot access the local TSQ service: {origin!r}."
        )
    host = hostname
    if ":" in host:
        host = f"[{host}]"
    default_port = 80 if parsed.scheme == "http" else 443
    port_suffix = "" if port in {None, default_port} else f":{port}"
    return f"{parsed.scheme}://{host}{port_suffix}"


def _loopback_host(host: str) -> bool:
    return host == DEFAULT_HOST


class ApiApplication:
    """Testable HTTP router backed by one TSQ database handle."""

    def __init__(
        self,
        database: Database,
        *,
        allowed_origins: Sequence[str] = DEFAULT_ALLOWED_ORIGINS,
    ) -> None:
        self.database = database
        self.engine = AdaptiveEngine(database)
        self.allowed_origins = frozenset(
            _validate_origin(origin) for origin in allowed_origins
        )

    def dispatch(
        self,
        method: str,
        target: str,
        *,
        headers: Mapping[str, str] | None = None,
        body: bytes = b"",
    ) -> ApiResponse:
        """Route one request and translate safe domain failures to JSON."""

        normalized_headers = {
            key.casefold(): value for key, value in (headers or {}).items()
        }
        origin = normalized_headers.get("origin")
        cors_headers: dict[str, str] = {}
        if origin is not None:
            try:
                normalized_origin = _validate_origin(origin)
            except ValueError:
                return self._error(
                    HTTPStatus.FORBIDDEN,
                    "origin_not_allowed",
                    "This origin is not allowed to call the local TSQ service.",
                )
            if normalized_origin not in self.allowed_origins:
                return self._error(
                    HTTPStatus.FORBIDDEN,
                    "origin_not_allowed",
                    "This origin is not allowed to call the local TSQ service.",
                )
            cors_headers = {
                "Access-Control-Allow-Origin": normalized_origin,
                "Vary": "Origin",
            }

        if method.upper() == "OPTIONS":
            response_headers = {
                **cors_headers,
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": (
                    "Content-Type, Idempotency-Key"
                ),
                "Access-Control-Max-Age": "600",
            }
            return ApiResponse(HTTPStatus.NO_CONTENT, None, response_headers)

        try:
            payload = self._parse_body(method, normalized_headers, body)
            result = self._route(
                method.upper(),
                target,
                normalized_headers,
                payload,
            )
            return ApiResponse(result.status, result.payload, {
                **result.headers,
                **cors_headers,
            })
        except MethodNotAllowedError as exc:
            response = ApiResponse(
                HTTPStatus.METHOD_NOT_ALLOWED,
                {
                    "error": {
                        "code": "method_not_allowed",
                        "message": str(exc),
                    }
                },
                {"Allow": ", ".join(exc.allowed)},
            )
        except ValidationError as exc:
            response = self._error(
                HTTPStatus.BAD_REQUEST, "validation_error", str(exc)
            )
        except NotFoundError as exc:
            response = self._error(
                HTTPStatus.NOT_FOUND, "not_found", str(exc)
            )
        except ExhaustedError as exc:
            response = self._error(
                HTTPStatus.CONFLICT, "corpus_exhausted", str(exc)
            )
        except ConflictError as exc:
            response = self._error(
                HTTPStatus.CONFLICT, "conflict", str(exc)
            )
        except TSQError as exc:
            response = self._error(
                HTTPStatus.BAD_REQUEST, "tsq_error", str(exc)
            )
        except sqlite3.Error:
            LOGGER.exception("The TSQ database request failed")
            response = self._error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "database_error",
                "The local TSQ database could not complete this request.",
            )
        except (OverflowError, TypeError, ValueError):
            response = self._error(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                "The request could not be interpreted.",
            )
        except Exception:
            LOGGER.exception("The TSQ API request failed unexpectedly")
            response = self._error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "internal_error",
                "The local TSQ service could not complete this request.",
            )
        return ApiResponse(response.status, response.payload, {
            **response.headers,
            **cors_headers,
        })

    @staticmethod
    def _error(status: int, code: str, message: str) -> ApiResponse:
        return ApiResponse(
            status,
            {"error": {"code": code, "message": message}},
        )

    @staticmethod
    def _parse_body(
        method: str,
        headers: Mapping[str, str],
        body: bytes,
    ) -> dict[str, Any]:
        if len(body) > MAX_REQUEST_BYTES:
            raise ValidationError(
                f"Request bodies are limited to {MAX_REQUEST_BYTES} bytes."
            )
        if method.upper() not in {"POST", "PUT", "PATCH"}:
            if body:
                raise ValidationError("GET and OPTIONS requests cannot have a body.")
            return {}
        if not body:
            return {}
        content_type = headers.get("content-type", "").split(";", 1)[0].strip()
        if content_type != "application/json":
            raise ValidationError("Request content type must be application/json.")
        try:
            decoded = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValidationError("Request body must be valid UTF-8 JSON.") from exc
        if type(decoded) is not dict:
            raise ValidationError("Request body must be a JSON object.")
        return decoded

    def _route(
        self,
        method: str,
        target: str,
        headers: Mapping[str, str],
        body: dict[str, Any],
    ) -> ApiResponse:
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or parsed.fragment:
            raise ValidationError("Request target must be a local path.")
        try:
            query = parse_qs(
                parsed.query,
                keep_blank_values=True,
                strict_parsing=False,
                max_num_fields=20,
            )
        except ValueError as exc:
            raise ValidationError("Request query is too large.") from exc
        raw_segments = [segment for segment in parsed.path.split("/") if segment]
        segments = tuple(unquote(segment) for segment in raw_segments)
        if any("/" in segment or "\x00" in segment for segment in segments):
            raise ValidationError("Path segments contain unsupported characters.")
        if segments[:2] == API_PREFIX:
            route = segments[2:]
        elif segments == ("health",):
            route = segments
        else:
            return self._error(
                HTTPStatus.NOT_FOUND,
                "route_not_found",
                "No API route matches this request.",
            )

        if route == ("health",):
            self._require_method(method, "GET")
            self._require_query(query, set())
            return ApiResponse(HTTPStatus.OK, self._health())
        if route == ("catalog",):
            self._require_method(method, "GET")
            self._require_query(query, set())
            return ApiResponse(HTTPStatus.OK, self._catalog())
        if route == ("topics",):
            self._require_method(method, "GET")
            self._require_query(query, set())
            catalog = self._catalog()
            return ApiResponse(
                HTTPStatus.OK,
                {
                    "release_id": catalog["release_id"],
                    "counts": catalog["counts"],
                    "topics": catalog["topics"],
                },
            )
        if len(route) == 2 and route[0] == "topics":
            self._require_method(method, "GET")
            self._require_query(query, set())
            return ApiResponse(HTTPStatus.OK, self._topic(route[1]))
        if route == ("learners",):
            self._require_method(method, "POST")
            self._require_query(query, set())
            self._require_fields(
                body,
                allowed={"learner_id", "display_name"},
                required={"learner_id"},
            )
            learner = self.engine.create_learner(
                body["learner_id"], body.get("display_name")
            )
            return ApiResponse(HTTPStatus.OK, learner)
        if len(route) == 2 and route[0] == "learners":
            self._require_method(method, "GET")
            self._require_query(query, set())
            return ApiResponse(HTTPStatus.OK, self._learner(route[1]))
        if len(route) == 3 and route[0] == "learners" and route[2] == "profile":
            self._require_method(method, "GET")
            self._require_query(query, {"topic_id"})
            topic_id = self._single_query(query, "topic_id")
            return ApiResponse(
                HTTPStatus.OK,
                self.engine.profile(route[1], root_concept_id=topic_id),
            )
        if route == ("sessions",):
            if method == "GET":
                self._require_fields(body, allowed=set())
                return ApiResponse(HTTPStatus.OK, self._list_sessions(query))
            self._require_method(method, "POST")
            self._require_query(query, set())
            self._require_fields(
                body,
                allowed={
                    "learner_id",
                    "topic_id",
                    "root_concept_id",
                    "explore_related",
                    "mode",
                    "seed",
                    "idempotency_key",
                },
                required={"learner_id"},
            )
            idempotency_key = self._idempotency_key(headers, body, required=True)
            explore_related = body.get("explore_related", True)
            if type(explore_related) is not bool:
                raise ValidationError("explore_related must be true or false.")
            session = self.engine.start_session(
                body["learner_id"],
                body.get("root_concept_id"),
                topic_id=body.get("topic_id"),
                explore_related=explore_related,
                mode=body.get("mode", "learn"),
                seed=body.get("seed"),
                idempotency_key=idempotency_key,
            )
            return ApiResponse(HTTPStatus.CREATED, session)
        if len(route) == 2 and route[0] == "sessions":
            self._require_method(method, "GET")
            self._require_query(query, set())
            return ApiResponse(
                HTTPStatus.OK, self.database.get_session(route[1])
            )
        if len(route) == 3 and route[0] == "sessions" and route[2] == "next":
            self._require_method(method, "POST")
            self._require_query(query, set())
            self._require_fields(body, allowed={"idempotency_key"})
            # AdaptivePolicy reuses its one durable pending presentation.  The
            # optional key is accepted for a uniform browser mutation contract;
            # the engine's persisted-pending boundary supplies the idempotence.
            self._idempotency_key(headers, body, required=False)
            return ApiResponse(
                HTTPStatus.OK,
                presentation_payload(self.engine.next_question(route[1])),
            )
        if len(route) == 3 and route[0] == "sessions" and route[2] == "end":
            self._require_method(method, "POST")
            self._require_query(query, set())
            self._require_fields(
                body,
                allowed={
                    "status",
                    "completed",
                    "reason",
                    "idempotency_key",
                },
            )
            idempotency_key = self._idempotency_key(headers, body, required=True)
            session = self.engine.end_session(
                route[1],
                status=body.get("status"),
                completed=body.get("completed"),
                reason=body.get("reason"),
                idempotency_key=idempotency_key,
            )
            return ApiResponse(HTTPStatus.OK, session)
        if len(route) == 3 and route[0] == "sessions" and route[2] == "report":
            self._require_method(method, "GET")
            self._require_query(query, set())
            return ApiResponse(
                HTTPStatus.OK, self.engine.session_report(route[1])
            )
        if len(route) == 3 and route[0] == "sessions" and route[2] == "trace":
            self._require_method(method, "GET")
            self._require_query(query, set())
            return ApiResponse(HTTPStatus.OK, self.engine.trace(route[1]))
        if len(route) == 3 and route[0] == "decisions" and route[2] == "answers":
            self._require_method(method, "POST")
            self._require_query(query, set())
            self._require_fields(
                body,
                allowed={
                    "option_id",
                    "confidence",
                    "response_ms",
                    "hint_count",
                    "idempotency_key",
                },
            )
            idempotency_key = self._idempotency_key(headers, body, required=True)
            result = self.engine.submit_answer(
                route[1],
                body.get("option_id"),
                confidence=body.get("confidence"),
                response_ms=body.get("response_ms"),
                hint_count=body.get("hint_count", 0),
                feedback_shown=False,
                idempotency_key=idempotency_key,
            )
            return ApiResponse(
                HTTPStatus.OK,
                submission_payload(
                    result,
                    _decision_objective_names(self.database, route[1]),
                ),
            )
        if len(route) == 3 and route[0] == "decisions" and route[2] == "feedback":
            self._require_method(method, "POST")
            self._require_query(query, set())
            self._require_fields(body, allowed={"idempotency_key"})
            idempotency_key = self._idempotency_key(headers, body, required=True)
            return ApiResponse(
                HTTPStatus.OK,
                self._record_feedback_shown(route[1], idempotency_key),
            )
        return self._error(
            HTTPStatus.NOT_FOUND,
            "route_not_found",
            "No API route matches this request.",
        )

    @staticmethod
    def _require_method(method: str, *allowed: str) -> None:
        if method not in allowed:
            raise MethodNotAllowedError(method, allowed)

    @staticmethod
    def _require_fields(
        body: Mapping[str, Any],
        *,
        allowed: set[str],
        required: set[str] | None = None,
    ) -> None:
        unknown = set(body) - allowed
        missing = (required or set()) - set(body)
        if unknown:
            raise ValidationError(
                "Unexpected request field(s): " + ", ".join(sorted(unknown)) + "."
            )
        if missing:
            raise ValidationError(
                "Missing request field(s): " + ", ".join(sorted(missing)) + "."
            )

    @staticmethod
    def _require_query(query: Mapping[str, list[str]], allowed: set[str]) -> None:
        unknown = set(query) - allowed
        if unknown:
            raise ValidationError(
                "Unexpected query field(s): " + ", ".join(sorted(unknown)) + "."
            )

    @staticmethod
    def _single_query(
        query: Mapping[str, list[str]], key: str
    ) -> str | None:
        values = query.get(key)
        if values is None:
            return None
        if len(values) != 1 or not values[0].strip():
            raise ValidationError(f"Query field {key} must have one non-blank value.")
        return values[0]

    @staticmethod
    def _idempotency_key(
        headers: Mapping[str, str],
        body: Mapping[str, Any],
        *,
        required: bool,
    ) -> str | None:
        header_key = headers.get("idempotency-key")
        body_key = body.get("idempotency_key")
        if header_key is not None and body_key is not None and header_key != body_key:
            raise ValidationError(
                "Idempotency-Key header and idempotency_key body field must match."
            )
        key = header_key if header_key is not None else body_key
        if key is None:
            if required:
                raise ValidationError(
                    "This mutation requires an Idempotency-Key header."
                )
            return None
        if not isinstance(key, str) or not key.strip() or len(key) > 200:
            raise ValidationError(
                "Idempotency-Key must be a non-blank string of at most 200 characters."
            )
        return key

    def _health(self) -> dict[str, Any]:
        release_id = self.database.get_active_release_id()
        counts = self._corpus_counts(release_id)
        return {
            "status": "ok",
            "api_version": API_VERSION,
            "schema_version": SCHEMA_VERSION,
            "corpus_release_id": release_id,
            "corpus": counts,
        }

    def _corpus_counts(self, release_id: str) -> dict[str, int]:
        with self.database.read() as connection:
            counts = connection.execute(
                """SELECT
                       (SELECT COUNT(*) FROM release_topics
                        WHERE release_id = ?) AS topics,
                       (SELECT COUNT(*) FROM release_concepts
                        WHERE release_id = ?) AS concepts,
                       (SELECT COUNT(*) FROM release_learning_objectives
                        WHERE release_id = ?) AS learning_objectives,
                       (SELECT COUNT(*) FROM release_questions
                        WHERE release_id = ?) AS questions,
                       (SELECT COUNT(*) FROM release_misconceptions
                        WHERE release_id = ?) AS misconceptions,
                       (SELECT COUNT(*) FROM release_sources
                        WHERE release_id = ?) AS sources,
                       (SELECT COUNT(*) FROM release_questions question
                        WHERE question.release_id = ?
                          AND question.status IN ('approved', 'calibrated')
                          AND NOT EXISTS (
                              SELECT 1 FROM question_revocations revoked
                              WHERE revoked.question_id = question.question_id
                          )) AS active_questions,
                       (SELECT COUNT(*) FROM release_questions question
                        WHERE question.release_id = ?
                          AND question.status = 'retired') AS retired_questions,
                       (SELECT COUNT(DISTINCT
                            tsq_canonical_family(registry.family_id))
                        FROM release_questions membership
                        JOIN questions registry
                          ON registry.id = membership.question_id
                        WHERE membership.release_id = ?
                          AND membership.status IN ('approved', 'calibrated')
                          AND NOT EXISTS (
                              SELECT 1 FROM question_revocations revoked
                              WHERE revoked.question_id = membership.question_id
                          )) AS active_families""",
                (release_id,) * 9,
            ).fetchone()
        assert counts is not None
        return {key: int(counts[key]) for key in counts.keys()}

    def _catalog(self) -> dict[str, Any]:
        """Return the release catalog with exact descendant-scope counts."""

        catalog = self.database.get_catalog()
        objectives = self.database.get_learning_objectives(catalog["release_id"])
        objective_count_by_concept: dict[str, int] = {}
        for objective in objectives:
            objective_count_by_concept[objective.primary_concept_id] = (
                objective_count_by_concept.get(objective.primary_concept_id, 0)
                + 1
            )
        children: dict[str | None, list[dict[str, Any]]] = {}
        for topic in catalog["topics"]:
            children.setdefault(topic["parent_id"], []).append(topic)
        for siblings in children.values():
            siblings.sort(
                key=lambda item: (item["sort_order"], item["name"], item["id"])
            )

        rows: list[dict[str, Any]] = []

        def append_topic(
            topic: dict[str, Any], depth: int, path: list[str]
        ) -> tuple[int, int, int]:
            direct_objectives = sum(
                objective_count_by_concept.get(concept["id"], 0)
                for concept in topic["concepts"]
            )
            scope_questions = int(topic["direct_primary_questions"])
            scope_concepts = len(topic["concepts"])
            scope_objectives = direct_objectives
            row = {
                **topic,
                "depth": depth,
                "path": [*path, topic["name"]],
                "direct_concepts": len(topic["concepts"]),
                "direct_learning_objectives": direct_objectives,
            }
            rows.append(row)
            for child in children.get(topic["id"], []):
                child_questions, child_concepts, child_objectives = append_topic(
                    child, depth + 1, row["path"]
                )
                scope_questions += child_questions
                scope_concepts += child_concepts
                scope_objectives += child_objectives
            row["scope_primary_questions"] = scope_questions
            row["scope_concepts"] = scope_concepts
            row["scope_learning_objectives"] = scope_objectives
            return scope_questions, scope_concepts, scope_objectives

        for domain in catalog["domains"]:
            for topic in children.get(None, []):
                if topic["domain_id"] == domain["id"]:
                    append_topic(topic, 0, [domain["name"]])
        return {
            "release_id": catalog["release_id"],
            "counts": self._corpus_counts(catalog["release_id"]),
            "domains": catalog["domains"],
            "topics": rows,
        }

    def _learner(self, learner_id: str) -> dict[str, Any]:
        with self.database.read() as connection:
            row = connection.execute(
                "SELECT * FROM learners WHERE id = ?", (learner_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"Unknown learner: {learner_id}")
        return dict(row)

    def _record_feedback_shown(
        self, decision_id: str, idempotency_key: str
    ) -> dict[str, Any]:
        """Record the same post-render feedback boundary used by the CLI."""

        with self.database.read() as connection:
            row = connection.execute(
                """SELECT attempt.outcome_json
                   FROM decisions decision
                   LEFT JOIN attempts attempt ON attempt.decision_id = decision.id
                   WHERE decision.id = ?""",
                (decision_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError(f"Unknown decision: {decision_id}")
        if not row["outcome_json"]:
            raise ConflictError(
                "Feedback cannot be shown before the decision has an answer."
            )
        outcome = json.loads(row["outcome_json"])
        selected = outcome.get("selected_option")
        correct = outcome.get("correct_option")
        if not isinstance(correct, dict):
            raise ConflictError(
                "The stored answer has no correct-option feedback; verify the database."
            )
        material = json.dumps(
            {
                "decision_id": decision_id,
                "selected_option_id": (
                    selected.get("id") if isinstance(selected, dict) else None
                ),
                "correct_option_id": correct.get("id"),
                "selected_rationale": (
                    selected.get("rationale")
                    if isinstance(selected, dict)
                    else None
                ),
                "correct_rationale": correct.get("rationale"),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        feedback_digest = hashlib.sha256(material).hexdigest()
        return self.engine.record_action(
            decision_id,
            "feedback_shown",
            {"feedback_digest": feedback_digest},
            stage="post_feedback",
            idempotency_key=idempotency_key,
        )

    def _topic(self, reference: str) -> dict[str, Any]:
        topic = self.database.resolve_topic(reference)
        scope = self.database.topic_scope(topic["id"], topic["release_id"])
        graph = self.database.get_graph(topic["release_id"])
        objectives = self.database.get_learning_objectives(
            topic["release_id"], primary_concept_ids=scope
        )
        return {
            "release_id": topic["release_id"],
            "topic": {
                key: value for key, value in topic.items() if key != "release_id"
            },
            "scope_concepts": [
                {
                    "id": concept_id,
                    "name": graph.concepts[concept_id].name,
                    "description": graph.concepts[concept_id].description,
                    "domain": graph.concepts[concept_id].domain,
                }
                for concept_id in sorted(
                    scope,
                    key=lambda item: (graph.concepts[item].name, item),
                )
            ],
            "learning_objectives": [
                {
                    "id": objective.id,
                    "name": objective.name,
                    "description": objective.description,
                    "primary_concept_id": objective.primary_concept_id,
                    "supporting_concept_ids": list(
                        objective.supporting_concept_ids
                    ),
                    "operation": objective.operation.value,
                    "evidence_type": objective.evidence_type,
                }
                for objective in objectives
            ],
        }

    def _list_sessions(self, query: Mapping[str, list[str]]) -> dict[str, Any]:
        self._require_query(query, {"learner_id", "status", "limit"})
        learner_id = self._single_query(query, "learner_id")
        status = self._single_query(query, "status")
        if status is not None and status not in {"active", "completed", "abandoned"}:
            raise ValidationError(
                "Session status must be active, completed, or abandoned."
            )
        raw_limit = self._single_query(query, "limit")
        try:
            limit = 50 if raw_limit is None else int(raw_limit)
        except ValueError as exc:
            raise ValidationError("Session limit must be an integer.") from exc
        if not 1 <= limit <= MAX_SESSION_LIST_LIMIT:
            raise ValidationError(
                f"Session limit must be from 1 to {MAX_SESSION_LIST_LIMIT}."
            )
        filters: list[str] = []
        parameters: list[Any] = []
        if learner_id is not None:
            filters.append("session_row.learner_id = ?")
            parameters.append(learner_id)
        if status is not None:
            filters.append("session_row.status = ?")
            parameters.append(status)
        where = "WHERE " + " AND ".join(filters) if filters else ""
        parameters.append(limit)
        with self.database.read() as connection:
            rows = connection.execute(
                f"""WITH selected_sessions AS MATERIALIZED (
                        SELECT session_row.*
                        FROM sessions session_row
                        {where}
                        ORDER BY session_row.updated_at DESC, session_row.id DESC
                        LIMIT ?
                    ), attempt_stats AS (
                        SELECT attempt.session_id,
                               COUNT(*) AS questions_answered,
                               SUM(attempt.is_correct) AS correct,
                               SUM(CASE WHEN attempt.selected_option_id IS NULL
                                        THEN 1 ELSE 0 END) AS abstained
                        FROM attempts attempt
                        JOIN selected_sessions selected
                          ON selected.id = attempt.session_id
                        GROUP BY attempt.session_id
                    )
                    SELECT session_row.id, session_row.learner_id,
                           learner.display_name AS learner_name,
                           session_row.corpus_release_id, session_row.topic_id,
                           session_row.root_concept_id,
                           COALESCE(topic.name, concept.name) AS target_name,
                           session_row.mode, session_row.phase, session_row.status,
                           session_row.step, session_row.created_at,
                           session_row.updated_at,
                           COALESCE(stats.questions_answered, 0)
                               AS questions_answered,
                           COALESCE(stats.correct, 0) AS correct,
                           COALESCE(stats.abstained, 0) AS abstained
                    FROM selected_sessions session_row
                    JOIN learners learner ON learner.id = session_row.learner_id
                    JOIN concepts concept ON concept.id = session_row.root_concept_id
                    LEFT JOIN release_topics topic
                      ON topic.release_id = session_row.corpus_release_id
                     AND topic.topic_id = session_row.topic_id
                    LEFT JOIN attempt_stats stats ON stats.session_id = session_row.id
                    ORDER BY session_row.updated_at DESC, session_row.id DESC""",
                tuple(parameters),
            ).fetchall()
        sessions: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            answered = int(item["questions_answered"])
            correct = int(item["correct"])
            abstained = int(item["abstained"])
            selected_answers = answered - abstained
            item.update(
                {
                    "step": int(item["step"]),
                    "questions_answered": answered,
                    "correct": correct,
                    "abstained": abstained,
                    "accuracy": correct / answered if answered else None,
                    "selected_answers": selected_answers,
                    "selected_incorrect": selected_answers - correct,
                    "selected_accuracy": (
                        correct / selected_answers if selected_answers else None
                    ),
                }
            )
            sessions.append(item)
        return {"sessions": sessions, "limit": limit}


class ApiRequestHandler(BaseHTTPRequestHandler):
    """HTTP transport for :class:`ApiApplication`."""

    server_version = "TSQLocalApi/1"
    sys_version = ""

    def __init__(
        self,
        *args: Any,
        application: ApiApplication,
        **kwargs: Any,
    ) -> None:
        self.application = application
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
        self._handle("GET")

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler contract
        self._handle("POST")

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib handler contract
        self._handle("OPTIONS")

    def _handle(self, method: str) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send(
                ApiApplication._error(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_content_length",
                    "Content-Length must be an integer.",
                )
            )
            return
        if content_length < 0 or content_length > MAX_REQUEST_BYTES:
            self._send(
                ApiApplication._error(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    "request_too_large",
                    f"Request bodies are limited to {MAX_REQUEST_BYTES} bytes.",
                )
            )
            return
        body = self.rfile.read(content_length) if content_length else b""
        response = self.application.dispatch(
            method,
            self.path,
            headers={key: value for key, value in self.headers.items()},
            body=body,
        )
        self._send(response)

    def _send(self, response: ApiResponse) -> None:
        material = (
            b""
            if response.payload is None
            else json.dumps(
                response.payload,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        self.send_response(int(response.status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(material)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in response.headers.items():
            self.send_header(key, value)
        self.end_headers()
        if material:
            self.wfile.write(material)

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(
            "%s - - [%s] %s\n"
            % (self.address_string(), self.log_date_time_string(), format % args)
        )


def prepare_database(path: Path, *, corpus: Path | None = None) -> Database:
    """Open a current database and install the canonical corpus when needed."""

    database = Database(path)
    database.initialize()
    Database(path, read_only=True).validate_current_schema()
    if corpus is None:
        _ensure_starter_corpus(database)
    else:
        database.import_corpus(*read_and_parse(corpus, include_catalog=True))
    Database(path, read_only=True).validate_current_schema()
    database.get_active_release_id()
    return database


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tsq-api",
        description="Serve the local TSQ adaptive engine over a loopback JSON API.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(os.environ.get("TSQ_DB", Path.cwd() / "tsq.db")),
        help="SQLite database path (default: TSQ_DB or ./tsq.db)",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        help="Explicit corpus manifest/directory; defaults to TSQ's bundled corpus",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--allow-origin",
        action="append",
        dest="allowed_origins",
        help=(
            "Exact local web origin allowed by CORS; repeat for multiple origins "
            "(defaults to localhost and loopback on port 3000)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not _loopback_host(args.host):
        print(
            "tsq-api has no remote authentication and only permits "
            "--host 127.0.0.1.",
            file=sys.stderr,
        )
        return 2
    if not 1 <= args.port <= 65535:
        print("--port must be from 1 to 65535.", file=sys.stderr)
        return 2
    try:
        allowed_origins = tuple(args.allowed_origins or DEFAULT_ALLOWED_ORIGINS)
        application = ApiApplication(
            prepare_database(args.db, corpus=args.corpus),
            allowed_origins=allowed_origins,
        )
        handler = partial(ApiRequestHandler, application=application)
        server = ThreadingHTTPServer((args.host, args.port), handler)
        server.daemon_threads = True
    except (OSError, TSQError, ValueError) as exc:
        print(f"Could not start tsq-api: {exc}", file=sys.stderr)
        return 2
    print(
        f"TSQ API ready at http://{args.host}:{args.port}/api/{API_VERSION} "
        f"using {args.db}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nTSQ API stopped.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
