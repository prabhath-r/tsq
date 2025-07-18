#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0

"""Check or synchronize the curriculum and its installed-package resource."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "corpus" / "ai_curriculum.json"
PACKAGED = ROOT / "src" / "tsq" / "data" / "ai_curriculum.json"


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Replace the packaged resource with the canonical corpus bytes.",
    )
    args = parser.parse_args(argv)

    source_bytes = SOURCE.read_bytes()
    if args.write:
        PACKAGED.parent.mkdir(parents=True, exist_ok=True)
        PACKAGED.write_bytes(source_bytes)

    if not PACKAGED.is_file():
        print(f"missing packaged corpus: {PACKAGED}", file=sys.stderr)
        return 1
    packaged_bytes = PACKAGED.read_bytes()
    source_hash = _digest(source_bytes)
    packaged_hash = _digest(packaged_bytes)
    if source_hash != packaged_hash:
        print("curriculum copies differ", file=sys.stderr)
        print(f"  source:   {source_hash}", file=sys.stderr)
        print(f"  packaged: {packaged_hash}", file=sys.stderr)
        print("run this command with --write to synchronize them", file=sys.stderr)
        return 1
    print(f"curriculum synchronized: {source_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
