# SPDX-License-Identifier: MPL-2.0

"""Fail-closed provenance rules for activating generated questions.

Generation and model review can produce a quarantined artifact, but neither is
an activation authority. A generated question pinned as ``approved`` or
``calibrated`` must carry an immutable human-review commitment in its content
provenance. Since question provenance participates in the immutable content
hash and status participates in the release manifest, activation then requires
an explicit reviewed artifact and a newly sealed corpus release; changing only
the status of an unreviewed generated item remains invalid.

The commitment is intentionally small and syntactic. It records who attested,
when, and that the reviewer claims independence from the author. Validation
cannot establish that the attestation is truthful, so callers must still
verify the external human-review process.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import hashlib
from importlib.resources import files
import json
import re
from typing import Iterable


ACTIVE_QUESTION_STATUSES = frozenset({"approved", "calibrated"})
PUBLIC_QUESTION_IDENTITY_FIELDS = frozenset(
    {"provider", "model", "generator", "provider_name", "model_name"}
)
_PUBLIC_IDENTITY_KEY_SAFE_EXCEPTIONS = frozenset(
    {
        "generator_declared_provenance_sha256",
        "generator_output_sha256",
        "generator_provenance_sha256",
        "independent_model_review_count",
        "stripped_generator_authority_field_count",
    }
)
_PUBLIC_IDENTITY_KEY_TOKENS = frozenset(
    {"backend", "engine", "generator", "llm", "model", "provider", "vendor"}
)
_FIELD_TOKEN_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|[^A-Za-z0-9]+")
ACTIVATION_REVIEW_FIELD = "activation_review"
ACTIVATION_REVIEW_FIELDS = frozenset(
    {
        "reviewer_kind",
        "reviewer_id",
        "reviewed_at",
        "independent_of_author",
        "attestation_digest",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
LEGACY_UNATTESTED_COHORT_SCHEMA = "tsq-legacy-unattested-question-cohort-v1"
LEGACY_UNATTESTED_COHORT_SHA256 = (
    "d51b5e655160941b72fbb41c5667bac47c1654658ada4dc2168003c200def58c"
)
LEGACY_UNATTESTED_MANIFEST_SCHEMA = (
    "tsq-legacy-unattested-question-manifest-v1"
)
LEGACY_UNATTESTED_MANIFEST_RESOURCE = (
    "legacy_unattested_question_manifest.json"
)


@dataclass(frozen=True, slots=True)
class ProvenanceIssue:
    code: str
    field: str
    message: str


def _public_identity_key_forbidden(field: object) -> bool:
    """Return whether a provenance key can encode a model/vendor identity.

    Public provenance stays extensible, so this is deliberately narrower than
    a closed schema. It nevertheless examines normalized tokens rather than
    five literal top-level spellings, closing nested and alias forms such as
    ``metadata.provider``, ``modelName``, and ``generator_identity``. The few
    operational commitment/count fields below are safe because they reveal no
    identity value.
    """

    if type(field) is not str:
        return False
    normalized = field.strip()
    if not normalized:
        return False
    if normalized in _PUBLIC_IDENTITY_KEY_SAFE_EXCEPTIONS:
        return False
    token_list = [
        token.lower()
        for token in _FIELD_TOKEN_BOUNDARY.split(normalized)
        if token
    ]
    tokens = set(token_list)
    compact = "".join(token_list)
    if tokens & _PUBLIC_IDENTITY_KEY_TOKENS:
        return True
    if compact in {
        "provider",
        "providerid",
        "provideridentity",
        "providername",
        "providerversion",
        "vendor",
        "vendorid",
        "vendoridentity",
        "vendorname",
        "vendorversion",
    }:
        return True
    return compact in {
        "backendid",
        "backendidentity",
        "backendname",
        "backendversion",
        "engineid",
        "engineidentity",
        "enginename",
        "engineversion",
        "generatorid",
        "generatoridentity",
        "generatorname",
        "generatormodel",
        "generatorprovider",
        "llmid",
        "llmidentity",
        "llmname",
        "llmversion",
        "modelid",
        "modelidentity",
        "modelname",
        "modelprovider",
        "modelreviewer",
        "modelversion",
    }


def public_question_identity_paths(value: object) -> tuple[str, ...]:
    """Find identity-bearing field names recursively in public provenance."""

    paths: list[str] = []

    def visit(node: object, path: tuple[str, ...]) -> None:
        if type(node) is dict:
            for raw_key, child in node.items():
                key = str(raw_key)
                child_path = (*path, key)
                if _public_identity_key_forbidden(raw_key):
                    paths.append(".".join(child_path))
                visit(child, child_path)
        elif type(node) is list:
            for index, child in enumerate(node):
                visit(child, (*path, f"[{index}]"))

    visit(value, ())
    return tuple(sorted(set(paths)))


def strip_public_question_identity_fields(
    provenance: dict[str, object],
) -> tuple[dict[str, object], tuple[str, ...]]:
    """Copy provenance while removing identity-bearing keys at every depth."""

    removed: list[str] = []

    def copy_without_identity(node: object, path: tuple[str, ...]) -> object:
        if type(node) is dict:
            cleaned: dict[str, object] = {}
            for raw_key, child in node.items():
                key = str(raw_key)
                child_path = (*path, key)
                if _public_identity_key_forbidden(raw_key):
                    removed.append(".".join(child_path))
                    continue
                cleaned[key] = copy_without_identity(child, child_path)
            return cleaned
        if type(node) is list:
            return [
                copy_without_identity(child, (*path, f"[{index}]"))
                for index, child in enumerate(node)
            ]
        return node

    cleaned = copy_without_identity(provenance, ())
    assert type(cleaned) is dict
    return cleaned, tuple(sorted(set(removed)))


def legacy_question_identity_payload(
    *,
    question_id: object,
    version: object,
    family_id: object,
    stem: object,
    kind: object,
    difficulty: object,
    discrimination: object,
    guess_rate: object,
    slip_rate: object,
    concepts: Iterable[tuple[object, object, object]],
    options: Iterable[tuple[object, object, object, object, object, object]],
    source_ids: Iterable[object],
    provenance: object,
    tags: Iterable[object],
    revision_of: object,
    learning_objective_id: object,
) -> dict[str, object]:
    """Build the status-independent identity committed by the legacy gate.

    The payload deliberately normalizes both raw JSON rows and typed
    :class:`Question` objects into one representation. Lifecycle status is not
    content: an immutable legacy question may be safely quarantined without
    invalidating its identity. Objective and option-diagnostic bindings are
    included because they affect what a response claims to measure.
    """

    return {
        "id": question_id,
        "version": int(version),
        "family_id": family_id,
        "stem": stem,
        "kind": kind,
        "difficulty": float(difficulty),
        "discrimination": float(discrimination),
        "guess_rate": float(guess_rate),
        "slip_rate": float(slip_rate),
        "concepts": [
            {
                "concept_id": concept_id,
                "weight": float(weight),
                "role": role,
            }
            for concept_id, weight, role in concepts
        ],
        "options": [
            {
                "id": option_id,
                "text": text,
                "correct": correct,
                "rationale": rationale,
                "misconception_id": misconception_id,
                "diagnostic_objective_id": diagnostic_objective_id,
            }
            for (
                option_id,
                text,
                correct,
                rationale,
                misconception_id,
                diagnostic_objective_id,
            ) in options
        ],
        "source_ids": list(source_ids),
        "provenance": provenance,
        "tags": list(tags),
        "revision_of": revision_of,
        "learning_objective_id": learning_objective_id,
    }


def legacy_unattested_cohort_digest(
    question_payloads: Iterable[dict[str, object]],
) -> str:
    """Commit to the complete missing-marker cohort and every member's content."""

    members: list[tuple[str, str]] = []
    for payload in question_payloads:
        question_id = payload.get("id")
        if type(question_id) is not str or not question_id:
            raise ValueError("Legacy question identities require a non-blank ID.")
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        members.append((question_id, hashlib.sha256(encoded).hexdigest()))
    envelope = {
        "schema": LEGACY_UNATTESTED_COHORT_SCHEMA,
        "members": sorted(members),
    }
    encoded = json.dumps(
        envelope,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _legacy_identity_digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=1)
def _legacy_unattested_manifest() -> dict[str, str]:
    """Load and self-authenticate the exact legacy-member allowlist.

    The original cohort commitment remains the root of trust.  Materializing
    its member digests lets a later release retire an old item without forcing
    every other legacy item to acquire provenance that did not exist when it
    was published.  It does not permit a mutation or a new missing-marker ID.
    """

    try:
        raw = json.loads(
            files("tsq.data")
            .joinpath(LEGACY_UNATTESTED_MANIFEST_RESOURCE)
            .read_text(encoding="utf-8")
        )
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Legacy unattested-question manifest is unavailable or invalid."
        ) from exc
    if type(raw) is not dict or set(raw) != {
        "schema",
        "cohort_sha256",
        "members",
    }:
        raise ValueError(
            "Legacy unattested-question manifest has an invalid envelope."
        )
    if (
        raw["schema"] != LEGACY_UNATTESTED_MANIFEST_SCHEMA
        or raw["cohort_sha256"] != LEGACY_UNATTESTED_COHORT_SHA256
        or type(raw["members"]) is not dict
    ):
        raise ValueError(
            "Legacy unattested-question manifest has an invalid commitment."
        )
    members: dict[str, str] = {}
    for question_id, digest in raw["members"].items():
        if (
            type(question_id) is not str
            or not question_id
            or type(digest) is not str
            or _SHA256.fullmatch(digest) is None
        ):
            raise ValueError(
                "Legacy unattested-question manifest has an invalid member."
            )
        members[question_id] = digest
    envelope = {
        "schema": LEGACY_UNATTESTED_COHORT_SCHEMA,
        "members": sorted(members.items()),
    }
    encoded = json.dumps(
        envelope,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != LEGACY_UNATTESTED_COHORT_SHA256:
        raise ValueError(
            "Legacy unattested-question manifest does not match its cohort root."
        )
    return members


def legacy_unattested_member_compatible(
    payload: dict[str, object],
) -> bool:
    """Return whether one missing-marker identity is an exact legacy member."""

    try:
        question_id = payload.get("id")
        if type(question_id) is not str or not question_id:
            return False
        expected = _legacy_unattested_manifest().get(question_id)
        return expected is not None and _legacy_identity_digest(payload) == expected
    except (TypeError, ValueError):
        return False


def _aware_iso8601(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def question_provenance_issues(
    provenance: object,
    *,
    status: object,
    legacy_unattested_compatible: bool = False,
) -> tuple[ProvenanceIssue, ...]:
    """Validate the generated-question activation commitment.

    Quarantined generated artifacts need only retain the exact boolean
    ``generated`` marker. If ``human_review`` or ``activation_review`` claims
    are present, however, their types and fields are always validated. Active
    generated questions additionally require ``human_review is True`` and the
    complete strict review object below::

        {
          "reviewer_kind": "human",
          "reviewer_id": "stable-human-reviewer-id",
          "reviewed_at": "2026-07-23T12:00:00+00:00",
          "independent_of_author": true,
          "attestation_digest": "<64 lowercase hex characters>"
        }
    """

    if type(provenance) is not dict:
        return (
            ProvenanceIssue(
                "provenance_type",
                "",
                "Question provenance must be an object.",
            ),
        )

    issues: list[ProvenanceIssue] = []
    for field in public_question_identity_paths(provenance):
        issues.append(
            ProvenanceIssue(
                "public_provenance_identity_forbidden",
                field,
                "Public question provenance must not record vendor, model, "
                "or generator identity; keep operational identity in the "
                "generation-job ledger.",
            )
        )
    if "generated" not in provenance and not legacy_unattested_compatible:
        issues.append(
            ProvenanceIssue(
                "generated_provenance_required",
                "generated",
                "provenance.generated must be explicitly declared; only an "
                "exact immutable member of the committed legacy manifest may "
                "omit it.",
            )
        )
    generated = provenance.get("generated", False)
    if "generated" in provenance and type(generated) is not bool:
        issues.append(
            ProvenanceIssue(
                "generated_flag_type",
                "generated",
                "provenance.generated must be a boolean.",
            )
        )

    human_review = provenance.get("human_review")
    if "human_review" in provenance and type(human_review) is not bool:
        issues.append(
            ProvenanceIssue(
                "human_review_flag_type",
                "human_review",
                "provenance.human_review must be a boolean.",
            )
        )

    active_generated = (
        generated is True
        and type(status) is str
        and status in ACTIVE_QUESTION_STATUSES
    )
    if active_generated and human_review is not True:
        issues.append(
            ProvenanceIssue(
                "generated_human_review_required",
                "human_review",
                "Active generated questions require human_review=true.",
            )
        )

    review = provenance.get(ACTIVATION_REVIEW_FIELD)
    if review is None:
        if active_generated:
            issues.append(
                ProvenanceIssue(
                    "activation_review_required",
                    ACTIVATION_REVIEW_FIELD,
                    "Active generated questions require activation_review.",
                )
            )
        return tuple(issues)
    if type(review) is not dict:
        issues.append(
            ProvenanceIssue(
                "activation_review_type",
                ACTIVATION_REVIEW_FIELD,
                "provenance.activation_review must be an object.",
            )
        )
        return tuple(issues)

    fields = set(review)
    missing = sorted(ACTIVATION_REVIEW_FIELDS - fields)
    extra = sorted(fields - ACTIVATION_REVIEW_FIELDS)
    if missing:
        issues.append(
            ProvenanceIssue(
                "activation_review_missing_fields",
                ACTIVATION_REVIEW_FIELD,
                "provenance.activation_review is missing fields: "
                + ", ".join(missing),
            )
        )
    if extra:
        issues.append(
            ProvenanceIssue(
                "activation_review_extra_fields",
                ACTIVATION_REVIEW_FIELD,
                "provenance.activation_review has unsupported fields: "
                + ", ".join(extra),
            )
        )

    reviewer_kind = review.get("reviewer_kind")
    if reviewer_kind != "human":
        issues.append(
            ProvenanceIssue(
                "activation_review_reviewer_kind",
                f"{ACTIVATION_REVIEW_FIELD}.reviewer_kind",
                "provenance.activation_review.reviewer_kind must be 'human'.",
            )
        )
    reviewer_id = review.get("reviewer_id")
    if type(reviewer_id) is not str or not reviewer_id.strip():
        issues.append(
            ProvenanceIssue(
                "activation_review_reviewer_id",
                f"{ACTIVATION_REVIEW_FIELD}.reviewer_id",
                "provenance.activation_review.reviewer_id must be a non-blank string.",
            )
        )
    reviewed_at = review.get("reviewed_at")
    if (
        type(reviewed_at) is not str
        or not reviewed_at.strip()
        or not _aware_iso8601(reviewed_at)
    ):
        issues.append(
            ProvenanceIssue(
                "activation_review_timestamp",
                f"{ACTIVATION_REVIEW_FIELD}.reviewed_at",
                "provenance.activation_review.reviewed_at must be an aware ISO-8601 timestamp.",
            )
        )
    if review.get("independent_of_author") is not True:
        issues.append(
            ProvenanceIssue(
                "activation_review_independence",
                f"{ACTIVATION_REVIEW_FIELD}.independent_of_author",
                "provenance.activation_review.independent_of_author must be true.",
            )
        )
    digest = review.get("attestation_digest")
    if type(digest) is not str or _SHA256.fullmatch(digest) is None:
        issues.append(
            ProvenanceIssue(
                "activation_review_digest",
                f"{ACTIVATION_REVIEW_FIELD}.attestation_digest",
                "provenance.activation_review.attestation_digest must be a lowercase SHA-256 digest.",
            )
        )
    return tuple(issues)


def generated_question_runtime_safe(
    provenance: object,
    *,
    status: str,
) -> bool:
    """Fail closed for explicitly generated content already stored in SQLite.

    Exact legacy questions can predate the ``generated`` marker, so an absent
    marker remains a compatibility case whose identity is enforced at import.
    Once the marker is present it must be an exact boolean. Generated active
    content must satisfy the complete human activation-review commitment even
    when it came from a release created by an older TSQ binary.
    """

    if type(provenance) is not dict:
        return False
    if "generated" not in provenance:
        return True
    generated = provenance["generated"]
    if generated is False:
        return True
    if generated is not True:
        return False
    return not question_provenance_issues(
        provenance,
        status=status,
        legacy_unattested_compatible=True,
    )
