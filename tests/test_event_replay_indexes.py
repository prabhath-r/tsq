# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tsq.errors import ConflictError
from tsq.store import (
    SCHEMA_VERSION,
    Database,
    _capture_current_schema_contract,
    _expected_current_schema_contract,
    _expected_v20_schema_contract,
)

from tests.schema_upgrade_helpers import durable_database_fingerprint


_TRACE_EVENT_TYPES = (
    "PerformanceTaskStarted",
    "PerformanceActionRecorded",
    "PerformanceArtifactRunClaimed",
    "PerformanceArtifactRunObserved",
)
_TRACE_EVENT_TYPES_SQL = ", ".join(
    f"'{event_type}'" for event_type in _TRACE_EVENT_TYPES
)
_ARTIFACT_EVENT_TYPES_SQL = ", ".join(
    (
        "'PerformanceArtifactRunClaimed'",
        "'PerformanceArtifactRunObserved'",
    )
)
_TRACE_PARTIAL_PREDICATE = (
    "WHERE event_type IN ( " + _TRACE_EVENT_TYPES_SQL + " )"
)
_ARTIFACT_PARTIAL_PREDICATE = (
    "WHERE event_type IN ( " + _ARTIFACT_EVENT_TYPES_SQL + " )"
)
_V21_INDEX_NAMES = {
    "idx_events_causation_stream",
    "idx_events_correlation_stream",
    "idx_events_session_stream",
    "idx_events_stream_type_version",
    "idx_events_action_id_stream",
    "idx_events_action_trace_stream",
    "idx_events_artifact_action_stream",
    "idx_events_claim_caller_key_stream",
    "idx_events_claim_command_hash_stream",
    "idx_events_claim_request_digest_stream",
    "idx_events_claim_request_run_stream",
    "idx_events_check_action_stream",
    "idx_events_claim_payload_stream",
    "idx_events_observed_request_digest_stream",
    "idx_events_observed_request_run_stream",
    "idx_events_observed_result_digest_stream",
    "idx_events_performance_attempt_payload",
    "idx_events_payload_session_stream",
    "idx_events_receipt_artifact_action_stream",
    "idx_events_receipt_attempt_stream",
    "idx_events_receipt_claim_stream",
    "idx_events_receipt_digest_stream",
    "idx_events_receipt_id_stream",
    "idx_events_receipt_request_digest_stream",
    "idx_events_receipt_result_digest_stream",
}
_PERFORMANCE_PROJECTION_TABLES = (
    "performance_attempts",
    "performance_actions",
    "performance_artifact_run_claims",
    "performance_artifact_run_receipts",
    "performance_scoring_claims",
    "performance_scoring_reconciliations",
    "task_evaluations",
    "shadow_evidence_bundles",
)


def _schema_contract(database: Database):
    with database.read() as connection:
        return _capture_current_schema_contract(connection)


def _downgrade_to_exact_v20(database: Database) -> None:
    with database.transaction() as connection:
        database._downgrade_v21_contract_to_v20(connection)
        connection.execute(
            """UPDATE meta SET value='20'
               WHERE key='schema_version'"""
        )
    if _schema_contract(database) != _expected_v20_schema_contract():
        raise AssertionError(
            "Test fixture does not match the exact supported v20 contract."
        )


def _event_snapshot(database: Database) -> tuple[tuple[object, ...], ...]:
    with database.read() as connection:
        return tuple(
            tuple(row)
            for row in connection.execute(
                """SELECT * FROM events
                   ORDER BY stream_id, stream_version"""
            )
        )


def _projection_counts(database: Database) -> tuple[int, ...]:
    with database.read() as connection:
        return tuple(
            connection.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
            for table in _PERFORMANCE_PROJECTION_TABLES
        )


class EventReplayIndexTests(unittest.TestCase):
    def test_fresh_schema_installs_exact_indexes_used_by_scoped_queries(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "fresh-current.db")
            database.initialize()

            self.assertEqual(SCHEMA_VERSION, 22)
            self.assertEqual(
                _schema_contract(database),
                _expected_current_schema_contract(),
            )
            with database.read() as connection:
                indexes = {
                    row["name"]: row
                    for row in connection.execute(
                        "PRAGMA index_list('events')"
                    )
                    if row["name"] in _V21_INDEX_NAMES
                }
                index_sql = {
                    row["name"]: " ".join(row["sql"].split())
                    for row in connection.execute(
                        """SELECT name, sql FROM sqlite_master
                           WHERE type='index'"""
                    )
                    if row["name"] in _V21_INDEX_NAMES
                }
                self.assertEqual(set(indexes), _V21_INDEX_NAMES)
                self.assertEqual(set(index_sql), _V21_INDEX_NAMES)
                self.assertTrue(
                    indexes[
                        "idx_events_performance_attempt_payload"
                    ]["partial"]
                )
                self.assertTrue(
                    indexes[
                        "idx_events_action_trace_stream"
                    ]["partial"]
                )
                self.assertTrue(
                    indexes["idx_events_action_id_stream"]["partial"]
                )
                self.assertTrue(
                    indexes["idx_events_artifact_action_stream"]["partial"]
                )
                self.assertTrue(
                    indexes["idx_events_check_action_stream"]["partial"]
                )
                self.assertTrue(
                    indexes["idx_events_claim_payload_stream"]["partial"]
                )
                for index_name in (
                    "idx_events_claim_caller_key_stream",
                    "idx_events_claim_command_hash_stream",
                    "idx_events_claim_request_digest_stream",
                    "idx_events_claim_request_run_stream",
                    "idx_events_observed_request_digest_stream",
                    "idx_events_observed_request_run_stream",
                    "idx_events_observed_result_digest_stream",
                    "idx_events_receipt_digest_stream",
                    "idx_events_receipt_id_stream",
                    "idx_events_receipt_request_digest_stream",
                    "idx_events_receipt_result_digest_stream",
                ):
                    self.assertTrue(indexes[index_name]["partial"])
                self.assertTrue(
                    indexes[
                        "idx_events_receipt_artifact_action_stream"
                    ]["partial"]
                )
                self.assertTrue(
                    indexes["idx_events_receipt_attempt_stream"]["partial"]
                )
                self.assertTrue(
                    indexes["idx_events_receipt_claim_stream"]["partial"]
                )
                self.assertTrue(
                    indexes[
                        "idx_events_payload_session_stream"
                    ]["partial"]
                )
                self.assertTrue(
                    indexes["idx_events_causation_stream"]["partial"]
                )
                self.assertTrue(
                    indexes["idx_events_correlation_stream"]["partial"]
                )
                self.assertFalse(
                    indexes["idx_events_session_stream"]["partial"]
                )
                self.assertFalse(
                    indexes["idx_events_stream_type_version"]["partial"]
                )
                for index_name in (
                    "idx_events_causation_stream",
                    "idx_events_correlation_stream",
                    "idx_events_performance_attempt_payload",
                    "idx_events_payload_session_stream",
                ):
                    self.assertTrue(
                        index_sql[index_name].endswith(
                            _TRACE_PARTIAL_PREDICATE
                        ),
                        index_sql[index_name],
                    )
                for index_name in (
                    "idx_events_action_id_stream",
                    "idx_events_action_trace_stream",
                ):
                    self.assertTrue(
                        index_sql[index_name].endswith(
                            "WHERE event_type = "
                            "'PerformanceActionRecorded'"
                        ),
                        index_sql[index_name],
                    )
                for index_name in (
                    "idx_events_artifact_action_stream",
                    "idx_events_claim_caller_key_stream",
                    "idx_events_claim_command_hash_stream",
                    "idx_events_claim_request_digest_stream",
                    "idx_events_claim_request_run_stream",
                ):
                    self.assertTrue(
                        index_sql[index_name].endswith(
                            "WHERE event_type = "
                            "'PerformanceArtifactRunClaimed'"
                        ),
                        index_sql[index_name],
                    )
                for index_name in (
                    "idx_events_check_action_stream",
                    "idx_events_observed_request_digest_stream",
                    "idx_events_observed_request_run_stream",
                    "idx_events_observed_result_digest_stream",
                    "idx_events_receipt_artifact_action_stream",
                    "idx_events_receipt_attempt_stream",
                    "idx_events_receipt_claim_stream",
                    "idx_events_receipt_digest_stream",
                    "idx_events_receipt_id_stream",
                    "idx_events_receipt_request_digest_stream",
                    "idx_events_receipt_result_digest_stream",
                ):
                    self.assertTrue(
                        index_sql[index_name].endswith(
                            "WHERE event_type = "
                            "'PerformanceArtifactRunObserved'"
                        ),
                        index_sql[index_name],
                    )
                self.assertTrue(
                    index_sql["idx_events_claim_payload_stream"].endswith(
                        _ARTIFACT_PARTIAL_PREDICATE
                    ),
                    index_sql["idx_events_claim_payload_stream"],
                )

                plans = {
                    "action_id_payload": connection.execute(
                        """EXPLAIN QUERY PLAN
                           SELECT event_id FROM events
                           WHERE event_type='PerformanceActionRecorded'
                             AND json_extract(
                                 payload_json, '$.action.id'
                             )=?
                           ORDER BY stream_id, stream_version""",
                        ("pact_index_probe",),
                    ).fetchall(),
                    "action_trace_payload": connection.execute(
                        """EXPLAIN QUERY PLAN
                           SELECT event_id FROM events
                           WHERE event_type='PerformanceActionRecorded'
                             AND json_extract(
                                 payload_json, '$.action.trace_id'
                             )=?
                           ORDER BY stream_id, stream_version""",
                        ("pta_index_probe",),
                    ).fetchall(),
                    "artifact_action_payload": connection.execute(
                        """EXPLAIN QUERY PLAN
                           SELECT event_id FROM events
                           WHERE event_type='PerformanceArtifactRunClaimed'
                             AND json_extract(
                                 payload_json, '$.artifact_action_id'
                             )=?
                           ORDER BY stream_id, stream_version""",
                        ("pact_index_probe",),
                    ).fetchall(),
                    "check_action_payload": connection.execute(
                        """EXPLAIN QUERY PLAN
                           SELECT event_id FROM events
                           WHERE event_type='PerformanceArtifactRunObserved'
                             AND json_extract(
                                 payload_json, '$.check_action_id'
                             )=?
                           ORDER BY stream_id, stream_version""",
                        ("pact_index_probe",),
                    ).fetchall(),
                    "claim_payload": connection.execute(
                        """EXPLAIN QUERY PLAN
                           SELECT event_id FROM events
                           WHERE event_type IN ("""
                        + _ARTIFACT_EVENT_TYPES_SQL
                        + """)
                             AND json_extract(
                                 payload_json, '$.claim_id'
                             )=?
                           ORDER BY stream_id, stream_version""",
                        ("parc_index_probe",),
                    ).fetchall(),
                    "causation": connection.execute(
                        """EXPLAIN QUERY PLAN
                           SELECT event_id FROM events
                           WHERE event_type IN ("""
                        + _TRACE_EVENT_TYPES_SQL
                        + """)
                             AND causation_id=?
                           ORDER BY stream_id, stream_version""",
                        ("evt_index_probe",),
                    ).fetchall(),
                    "correlation": connection.execute(
                        """EXPLAIN QUERY PLAN
                           SELECT event_id FROM events
                           WHERE event_type IN ("""
                        + _TRACE_EVENT_TYPES_SQL
                        + """)
                             AND correlation_id=?
                           ORDER BY stream_id, stream_version""",
                        ("pta_index_probe",),
                    ).fetchall(),
                    "session": connection.execute(
                        """EXPLAIN QUERY PLAN
                           SELECT event_id FROM events
                           WHERE event_type IN ("""
                        + _TRACE_EVENT_TYPES_SQL
                        + """)
                             AND session_id=?
                           ORDER BY stream_version""",
                        ("ses_index_probe",),
                    ).fetchall(),
                    "payload": connection.execute(
                        """EXPLAIN QUERY PLAN
                           SELECT event_id FROM events
                           WHERE event_type IN ("""
                        + _TRACE_EVENT_TYPES_SQL
                        + """)
                             AND json_extract(
                                 payload_json, '$.attempt_id'
                             )=?
                           ORDER BY stream_id, stream_version""",
                        ("pta_index_probe",),
                    ).fetchall(),
                    "payload_session_trace": connection.execute(
                        """EXPLAIN QUERY PLAN
                           SELECT event_id FROM events
                           WHERE event_type IN ("""
                        + _TRACE_EVENT_TYPES_SQL
                        + """)
                             AND json_extract(
                                 payload_json, '$.session_id'
                             )=?
                           ORDER BY stream_id, stream_version""",
                        ("ses_index_probe",),
                    ).fetchall(),
                    "payload_session_start": connection.execute(
                        """EXPLAIN QUERY PLAN
                           SELECT event_id FROM events
                           WHERE event_type='PerformanceTaskStarted'
                             AND event_type IN ("""
                        + _TRACE_EVENT_TYPES_SQL
                        + """)
                             AND json_extract(
                                 payload_json, '$.session_id'
                             )=?
                           ORDER BY stream_id, stream_version""",
                        ("ses_index_probe",),
                    ).fetchall(),
                    "receipt_artifact_action": connection.execute(
                        """EXPLAIN QUERY PLAN
                           SELECT event_id FROM events
                           WHERE event_type='PerformanceArtifactRunObserved'
                             AND json_extract(
                                 payload_json,
                                 '$.receipt.artifact_action_id'
                             )=?
                           ORDER BY stream_id, stream_version""",
                        ("pact_index_probe",),
                    ).fetchall(),
                    "receipt_attempt": connection.execute(
                        """EXPLAIN QUERY PLAN
                           SELECT event_id FROM events
                           WHERE event_type='PerformanceArtifactRunObserved'
                             AND json_extract(
                                 payload_json, '$.receipt.attempt_id'
                             )=?
                           ORDER BY stream_id, stream_version""",
                        ("pta_index_probe",),
                    ).fetchall(),
                    "receipt_claim": connection.execute(
                        """EXPLAIN QUERY PLAN
                           SELECT event_id FROM events
                           WHERE event_type='PerformanceArtifactRunObserved'
                             AND json_extract(
                                 payload_json, '$.receipt.claim_id'
                             )=?
                           ORDER BY stream_id, stream_version""",
                        ("parc_index_probe",),
                    ).fetchall(),
                    "stream_type": connection.execute(
                        """EXPLAIN QUERY PLAN
                           SELECT COUNT(*) FROM events
                           WHERE stream_id=? AND event_type=?
                             AND stream_version<?""",
                        (
                            "learner:index-probe",
                            "ResponseSubmitted",
                            100,
                        ),
                    ).fetchall(),
                }
                identity_plan_specs = {
                    "claim_caller_key": (
                        "PerformanceArtifactRunClaimed",
                        "$.caller_idempotency_key",
                        "idx_events_claim_caller_key_stream",
                    ),
                    "claim_command_hash": (
                        "PerformanceArtifactRunClaimed",
                        "$.command_hash",
                        "idx_events_claim_command_hash_stream",
                    ),
                    "claim_request_digest": (
                        "PerformanceArtifactRunClaimed",
                        "$.request_digest",
                        "idx_events_claim_request_digest_stream",
                    ),
                    "claim_request_run": (
                        "PerformanceArtifactRunClaimed",
                        "$.request.run_id",
                        "idx_events_claim_request_run_stream",
                    ),
                    "observed_request_digest": (
                        "PerformanceArtifactRunObserved",
                        "$.result.request_digest",
                        "idx_events_observed_request_digest_stream",
                    ),
                    "observed_request_run": (
                        "PerformanceArtifactRunObserved",
                        "$.result.request.run_id",
                        "idx_events_observed_request_run_stream",
                    ),
                    "observed_result_digest": (
                        "PerformanceArtifactRunObserved",
                        "$.result_digest",
                        "idx_events_observed_result_digest_stream",
                    ),
                    "receipt_digest": (
                        "PerformanceArtifactRunObserved",
                        "$.receipt_digest",
                        "idx_events_receipt_digest_stream",
                    ),
                    "receipt_id": (
                        "PerformanceArtifactRunObserved",
                        "$.receipt_id",
                        "idx_events_receipt_id_stream",
                    ),
                    "receipt_request_digest": (
                        "PerformanceArtifactRunObserved",
                        "$.receipt.request_digest",
                        "idx_events_receipt_request_digest_stream",
                    ),
                    "receipt_result_digest": (
                        "PerformanceArtifactRunObserved",
                        "$.receipt.result_digest",
                        "idx_events_receipt_result_digest_stream",
                    ),
                }
                for label, (
                    event_type,
                    json_path,
                    _expected_index,
                ) in identity_plan_specs.items():
                    plans[label] = connection.execute(
                        "EXPLAIN QUERY PLAN "
                        "SELECT event_id FROM events "
                        f"WHERE event_type='{event_type}' "
                        "AND json_extract(payload_json, "
                        f"'{json_path}')=? "
                        "ORDER BY stream_id, stream_version",
                        ("identity_index_probe",),
                    ).fetchall()
            for label, expected_index in (
                ("action_id_payload", "idx_events_action_id_stream"),
                (
                    "action_trace_payload",
                    "idx_events_action_trace_stream",
                ),
                (
                    "artifact_action_payload",
                    "idx_events_artifact_action_stream",
                ),
                (
                    "check_action_payload",
                    "idx_events_check_action_stream",
                ),
                ("claim_payload", "idx_events_claim_payload_stream"),
                ("causation", "idx_events_causation_stream"),
                ("correlation", "idx_events_correlation_stream"),
                ("session", "idx_events_session_stream"),
                (
                    "payload",
                    "idx_events_performance_attempt_payload",
                ),
                (
                    "payload_session_trace",
                    "idx_events_payload_session_stream",
                ),
                (
                    "payload_session_start",
                    "idx_events_payload_session_stream",
                ),
                (
                    "receipt_artifact_action",
                    "idx_events_receipt_artifact_action_stream",
                ),
                (
                    "receipt_attempt",
                    "idx_events_receipt_attempt_stream",
                ),
                (
                    "receipt_claim",
                    "idx_events_receipt_claim_stream",
                ),
                ("stream_type", "idx_events_stream_type_version"),
            ):
                details = " ".join(row["detail"] for row in plans[label])
                self.assertIn(expected_index, details)
                self.assertIn("SEARCH", details)
                self.assertNotIn("SCAN events", details)
            for label, (
                _event_type,
                _json_path,
                expected_index,
            ) in identity_plan_specs.items():
                details = " ".join(row["detail"] for row in plans[label])
                self.assertIn(expected_index, details)
                self.assertIn("SEARCH", details)
                self.assertNotIn("SCAN events", details)

    def test_exact_v20_upgrade_only_indexes_existing_history(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "upgrade-v20.db")
            database.initialize()
            with database.transaction() as connection:
                database.append_event(
                    connection,
                    stream_id="system:schema-v21-index-fixture",
                    event_type="EventReplayIndexFixture",
                    payload={"attempt_id": "pta_schema_v21_fixture"},
                    metadata={"fixture": True},
                    session_id="ses_schema_v21_fixture",
                    correlation_id="pta_schema_v21_fixture",
                )
            _downgrade_to_exact_v20(database)
            before_events = _event_snapshot(database)
            before_projections = _projection_counts(database)

            database.initialize()

            self.assertEqual(
                _schema_contract(database),
                _expected_current_schema_contract(),
            )
            self.assertEqual(_event_snapshot(database), before_events)
            self.assertEqual(
                _projection_counts(database),
                before_projections,
            )
            with database.read() as connection:
                self.assertEqual(
                    connection.execute(
                        """SELECT value FROM meta
                           WHERE key='schema_version'"""
                    ).fetchone()["value"],
                    str(SCHEMA_VERSION),
                )
                self.assertEqual(
                    connection.execute(
                        "PRAGMA foreign_key_check"
                    ).fetchall(),
                    [],
                )

    def test_v20_index_drift_is_rejected_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "drifted-v20.db"
            database = Database(path)
            database.initialize()
            _downgrade_to_exact_v20(database)
            with database.transaction() as connection:
                connection.execute(
                    """CREATE INDEX idx_events_correlation_stream
                       ON events(correlation_id)"""
                )
            before = durable_database_fingerprint(path)

            with self.assertRaisesRegex(
                ConflictError,
                "exact supported v21 migration source",
            ):
                database.initialize()

            self.assertEqual(durable_database_fingerprint(path), before)
            with database.read() as connection:
                self.assertEqual(
                    connection.execute(
                        """SELECT value FROM meta
                           WHERE key='schema_version'"""
                    ).fetchone()["value"],
                    "20",
                )
                columns = [
                    row["name"]
                    for row in connection.execute(
                        """PRAGMA index_info(
                               'idx_events_correlation_stream'
                           )"""
                    )
                ]
            self.assertEqual(columns, ["correlation_id"])


if __name__ == "__main__":
    unittest.main()
