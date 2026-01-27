# SPDX-License-Identifier: MPL-2.0

"""The Second Question: an explainable adaptive learning engine."""

from .engine import AdaptiveEngine
from .store import Database

__all__ = ["AdaptiveEngine", "Database"]
__version__ = "0.1.0"
