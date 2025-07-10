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


