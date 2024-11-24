"""Agori - A secure cognitive architecture framework
with domain-specific capabilities."""

from agori.decision_frameworks import AVAILABLE_FRAMEWORKS, NGTProcessor

from .core.db import WorkingMemory
from .hr.hiring.screening import CandidateScreening
from .utils.exceptions import (
    AgoriException,
    ConfigurationError,
    ProcessingError,
    SearchError,
)

__version__ = "0.1.1"

# Package metadata
__title__ = "agori"
__description__ = "AI-powered decision making frameworks"
__author__ = "Your Name"
__license__ = "MIT"

__all__ = [
    # Core components
    "WorkingMemory",
    "CandidateScreening",
    # Base exceptions
    "AgoriException",
    "ConfigurationError",
    "ProcessingError",
    "SearchError",
    # NGT Framework
    "NGTProcessor",
    "NGTExpert",
    "NGTIdea",
    "NGTResponse",
    "NGTError",
    "ExpertGenerationError",
    "IdeaGenerationError",
    "SynthesisError",
    # Delphi Framework
    # "DelphiProcessor",
    # "DelphiExpert",
    # "DelphiResponse",
    # "DelphiReport",
    # "DelphiError",
    # "DelphiConfig",
    # "RoundExecutionError",
    # "ConvergenceError",
    # "Opinion",
    # Framework registry
    "AVAILABLE_FRAMEWORKS",
]

# Default configuration
DEFAULT_CONFIG = {
    "max_workers": 3,
    "chunk_size": 31500,
    "chunk_overlap": 2000,
    "delphi_max_rounds": 3,
    "delphi_convergence_threshold": 0.8,
    "delphi_min_experts": 5,
    "delphi_max_experts": 10,
}
