#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0

"""Check or synchronize the curriculum and its installed-package resource."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "corpus"
PACKAGED = ROOT / "src" / "tsq" / "data" / "curriculum"
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tsq.corpus import corpus_source_digest, read_and_parse  # noqa: E402
from tsq.errors import TSQError  # noqa: E402
from tsq.graph import KnowledgeGraph  # noqa: E402
from tsq.quality import audit_corpus  # noqa: E402


def _inventory(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*.json"))
        if path.is_file()
    }


def _digest(inventory: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for relative, payload in sorted(inventory.items()):
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Synchronize packaged JSON resources from the canonical corpus tree.",
    )
    args = parser.parse_args(argv)

    try:
        # The packaged tree is trusted runtime input. Fully assemble and parse
        # the canonical source before --write is allowed to touch that copy.
        concepts, edges, misconceptions, _, questions, _, _ = read_and_parse(
            SOURCE,
            include_catalog=True,
        )
        strict_issues = audit_corpus(
            questions,
            expected_primary_concept_ids={
                mapping.concept_id
                for question in questions
                for mapping in question.concepts
            },
            knowledge_graph=KnowledgeGraph(concepts, edges),
            misconceptions=misconceptions,
        )
        if strict_issues:
            print(
                "invalid canonical corpus: strict audit reported "
                f"{len(strict_issues)} issue(s)",
                file=sys.stderr,
            )
            for issue in strict_issues[:20]:
                print(
                    f"  [{issue.severity}] {issue.code}: {issue.message}",
                    file=sys.stderr,
                )
            return 1
        source_digest = corpus_source_digest(SOURCE)
    except TSQError as exc:
        print(f"invalid canonical corpus: {exc}", file=sys.stderr)
        return 1

    source_inventory = _inventory(SOURCE)
    if args.write:
        PACKAGED.mkdir(parents=True, exist_ok=True)
        for relative, payload in source_inventory.items():
            target = PACKAGED / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        for stale in sorted(PACKAGED.rglob("*.json")):
            if stale.relative_to(PACKAGED).as_posix() not in source_inventory:
                stale.unlink()

    if not PACKAGED.is_dir():
        print(f"missing packaged corpus: {PACKAGED}", file=sys.stderr)
        return 1
    packaged_inventory = _inventory(PACKAGED)
    source_hash = _digest(source_inventory)
    packaged_hash = _digest(packaged_inventory)
    if source_inventory != packaged_inventory:
        print("curriculum copies differ", file=sys.stderr)
        print(f"  source:   {source_hash}", file=sys.stderr)
        print(f"  packaged: {packaged_hash}", file=sys.stderr)
        missing = sorted(source_inventory.keys() - packaged_inventory.keys())
        extra = sorted(packaged_inventory.keys() - source_inventory.keys())
        changed = sorted(
            relative
            for relative in source_inventory.keys() & packaged_inventory.keys()
            if source_inventory[relative] != packaged_inventory[relative]
        )
        if missing:
            print(f"  missing:  {', '.join(missing)}", file=sys.stderr)
        if extra:
            print(f"  extra:    {', '.join(extra)}", file=sys.stderr)
        if changed:
            print(f"  changed:  {', '.join(changed)}", file=sys.stderr)
        print("run this command with --write to synchronize them", file=sys.stderr)
        return 1
    print(f"curriculum synchronized: {source_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
