# agori/__init__.py (KEEP AS IS - NO CHANGES NEEDED)
"""Agori - A secure cognitive architecture framework
with domain-specific capabilities."""

from .core.db import WorkingMemory
from .hr.hiring.screening import CandidateScreening
from .utils.exceptions import (
    AgoriException,
    ConfigurationError,
    ProcessingError,
    SearchError,
)

__version__ = "0.1.1"
__all__ = [
    "WorkingMemory",
    "CandidateScreening",
    "AgoriException",
    "ConfigurationError",
    "ProcessingError",
    "SearchError",
]
